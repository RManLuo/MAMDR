import json
import os
import os.path as osp
import time

import tensorflow as tf
from tensorflow.python import train
from tensorflow.python.keras import callbacks, backend

from utils.auc import AUC


class BaseModel(object):
    def __init__(self, dataset, config):
        self.n_uid = dataset.n_uid
        self.n_pid = dataset.n_pid
        self.n_domain = dataset.n_domain
        self.dataset = dataset
        self.config = config
        self.model_config = config['model']
        self.train_config = config['train']

        self.checkpoint_path = osp.join(self.train_config['checkpoint_path'], self.model_config['name'],
                                        self.dataset.conf['name'], dataset.conf['domain_split_path'],
                                        time.strftime("%a-%b-%d-%H-%M-%S", time.localtime()),
                                        "model_parameters.h5")
        self.result_path = osp.join(self.train_config['result_save_path'], self.model_config['name'],
                                    dataset.conf['name'], dataset.conf['domain_split_path'])
        # Model
        self.model = self.build_model()

        # Build Early Stop
        self._build_early_stop()

    def build_model(self):
        raise NotImplementedError("You must implement build model")

    def train(self):
        raise NotImplementedError

    def separate_train_val_test(self, init_parms=True):
        '''
        Separate train for each domain
        :param init_parms: bool, set to false for finetune.
        :return:
        '''
        # unfreeze the graph
        graph = tf.get_default_graph()
        if graph.finalized:
            graph._unsafe_unfinalize()

        train_dataset = self.dataset.train_dataset
        val_dataset = self.dataset.val_dataset
        test_dataset = self.dataset.test_dataset

        domain_loss = {}
        domain_auc = {}
        all_loss = 0
        all_auc = 0
        verbose = 0
        if init_parms:
            verbose = 2
            backend.get_session().run(tf.global_variables_initializer())
        # Save init weight
        weights = self.model.get_weights()
        for domain_idx, train_d in train_dataset.items():
            if not init_parms:
                # Reset adam learning rate
                opt = train.GradientDescentOptimizer(learning_rate=self.train_config['learning_rate'])
                self.model.compile(loss=self.train_config['loss'], optimizer=opt,
                                   metrics=[AUC(num_thresholds=500, name="AUC")])
            self.model.set_weights(weights)
            # Train
            print("Train on domain: {}".format(domain_idx))
            if not osp.exists(os.path.dirname(self.checkpoint_path)):
                os.makedirs(os.path.dirname(self.checkpoint_path))
            chk_path = os.path.join(os.path.dirname(self.checkpoint_path), "domain_{}.h5".format(domain_idx))
            callback = [
                callbacks.EarlyStopping(monitor='val_AUC', patience=self.train_config['patience'], mode='max',
                                        min_delta=1e-4),
                callbacks.ModelCheckpoint(chk_path, monitor='val_AUC', save_best_only=True,
                                          save_weights_only=True, mode='max')]
            # Finetune
            self.model.fit(train_d['data'].repeat(), steps_per_epoch=train_d['n_step'], verbose=verbose,
                           callbacks=callback,
                           validation_data=val_dataset[domain_idx]['data'].repeat(),
                           validation_steps=val_dataset[domain_idx]['n_step'],
                           epochs=self.train_config['epoch'])
            # Test
            self.model.load_weights(chk_path)
            p_loss, p_auc = self.model.evaluate(
                test_dataset[domain_idx]['data'], steps=test_dataset[domain_idx]['n_step'],
                verbose=0)

            p_loss, p_auc = float(p_loss), float(p_auc)
            domain_loss[domain_idx] = p_loss
            domain_auc[domain_idx] = p_auc
            all_loss += p_loss
            all_auc += p_auc

        # Restore meta weight
        self.model.set_weights(weights)
        avg_loss = all_loss / len(domain_loss)
        avg_auc = all_auc / len(domain_auc)
        print("Loss: ", domain_loss)
        self._format_print_domain_metric("AUC", domain_auc)
        weighted_auc = self._weighted_auc("test", domain_auc)
        print("Overall {} Loss: {}, AUC: {}, Weighted AUC: {}".format("test", avg_loss, avg_auc, weighted_auc))
        return avg_loss, avg_auc, domain_loss, domain_auc

    def val_and_test(self, mode):
        '''
        Validation or test the model
        :param mode: string, "val" or "test"
        :return:
        '''
        if mode == "val":
            dataset = self.dataset.val_dataset
        elif mode == "test":
            dataset = self.dataset.test_dataset
            self.load_model(self.checkpoint_path)  # Load best model weights
        else:
            raise ValueError("Mode can be either val or test, not: {}".format(mode))

        domain_loss = {}
        domain_auc = {}
        all_loss = 0
        all_auc = 0

        for idx, d in dataset.items():
            p_loss, p_auc = self.model.evaluate(d['data'], steps=d['n_step'],
                                                verbose=0)
            p_loss, p_auc = float(p_loss), float(p_auc)
            domain_loss[idx] = p_loss
            domain_auc[idx] = p_auc
            all_loss += p_loss
            all_auc += p_auc
        avg_loss = all_loss / len(domain_loss)
        avg_auc = all_auc / len(domain_auc)
        print("Loss: ", domain_loss)
        self._format_print_domain_metric("AUC", domain_auc)
        weighted_auc = self._weighted_auc(mode, domain_auc)
        print("Overall {} Loss: {}, AUC: {}, Weighted AUC: {}".format(mode, avg_loss, avg_auc, weighted_auc))
        return avg_loss, avg_auc, domain_loss, domain_auc

    def _format_print_domain_metric(self, name, domain_metric):
        '''
        Print metrics for each domain
        :param name: string, metric name
        :param domain_metric: dict, key: domain index, value: domain metric value
        :return:
        '''
        print(f"{name}: ")
        for key, value in domain_metric.items():
            print(f"{key}: {value}")

    def _weighted_auc(self, mode, domain_auc):
        '''
        Calculate the weighted AUC
        :param mode:
        :param domain_auc:
        :return:
        '''
        data_info = self.dataset.dataset_info
        tag = 'n_train'
        if "val" in mode:
            tag = "n_val"
        elif "test" in mode:
            tag = "n_test"
        weighted_auc = 0
        total_num = 0
        for key, value in domain_auc.items():
            weighted_auc += data_info[key][tag] * value
            total_num += data_info[key][tag]
        return weighted_auc / total_num

    def save_model(self, path):
        self.model.save_weights(path)

    def load_model(self, path):
        self.model.load_weights(path)

    def save_result(self, avg_loss, avg_auc, domain_loss, domain_auc):
        result_folder_name = "loss_{:.3f}_auc_{:.3f}_{}".format(avg_loss, avg_auc,
                                                                time.strftime("%a-%b-%d-%H-%M-%S", time.localtime()))
        result_path = osp.join(self.result_path, result_folder_name)
        if not osp.exists(result_path):
            os.makedirs(result_path)
        with open(osp.join(result_path, "dataset_info.json"), 'w') as f:
            json.dump(self.dataset.dataset_info, f)
        with open(osp.join(result_path, "config.json.example"), 'w') as f:
            json.dump(self.config, f)
        with open(osp.join(result_path, "result.json"), 'w') as f:
            json.dump({
                "avg_loss": avg_loss,
                "avg_auc": avg_auc,
                "domain_loss": domain_loss,
                "domain_auc": domain_auc
            }, f)
        self.save_model(osp.join(result_path, "model_parameters.h5"))

    def _build_early_stop(self):
        self.patience = self.train_config['patience']
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def early_stop_step(self, metric):
        if not osp.exists(os.path.dirname(self.checkpoint_path)):
            os.makedirs(os.path.dirname(self.checkpoint_path))

        if self.best_metric is None:
            self.best_metric = metric
            self.save_model(self.checkpoint_path)
        elif (metric <= self.best_metric):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}, Best AUC: {self.best_metric}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_model(self.checkpoint_path)
            self.best_metric = metric
            self.counter = 0
        return self.early_stop
