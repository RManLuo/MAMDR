import os
import os.path as osp
import random
import time
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.python import train
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks, backend

from model_zoo import BaseModel
from utils.auc import AUC
from .maml import MAML
class SpecificBase(MAML):
    def __init__(self, base_model: BaseModel):
        super(SpecificBase, self).__init__(base_model)

    def build_meta_data_split(self):
        meta_data_split = {}
        meta_sequence = []

        if self.train_config['target_domain'] >= 0:
            target_domain = self.train_config['target_domain']
            target_data = self.dataset.train_dataset[target_domain]
            self.target_iter = target_data['data'].make_initializable_iterator()
            self.target_step = target_data['n_step']

        for idx, d in self.dataset.train_dataset.items():
            train_iter = d['data'].make_initializable_iterator()
            meta_data_split[idx] = {"train_iter": train_iter, "train_step": d['n_step']}
            # skip the target domain
            if self.train_config['target_domain'] >= 0 and idx in [self.train_config['target_domain']]:
                continue
            meta_sequence.append(idx)

        if "meta_sequence" in self.train_config and isinstance(self.train_config['meta_sequence'], list):
            if len(self.train_config['meta_sequence']) != len(meta_sequence):
                raise ValueError("All the domains must be given in the sequence")
            meta_sequence = self.train_config['meta_sequence']
        return meta_data_split, meta_sequence

    def early_stop_step(self, metric):

        if self.best_metric is None:
            self.best_metric = metric
            self.best_shared_weights = deepcopy(self.meta_weights)
            self.best_domain_weights = deepcopy(self.domain_weights)
            self.save_model(self.checkpoint_path)
        elif (metric <= self.best_metric):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}, Best AUC: {self.best_metric}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_model(self.checkpoint_path)
            self.best_metric = metric
            self.best_shared_weights = deepcopy(self.meta_weights)
            self.best_domain_weights = deepcopy(self.domain_weights)
            self.counter = 0
        return self.early_stop

    def val_and_test(self, mode):
        if mode == "val":
            dataset = self.dataset.val_dataset
            val_test_shared_weights = deepcopy(self.meta_weights)
            val_test_domain_weights = deepcopy(self.domain_weights)
        elif mode == "test":
            dataset = self.dataset.test_dataset
            self.load_model(self.checkpoint_path)
            val_test_shared_weights = deepcopy(self.best_shared_weights)
            val_test_domain_weights = deepcopy(self.best_domain_weights)
        else:
            raise ValueError("Mode can be either val or test, not: {}".format(mode))

        domain_loss = {}
        domain_auc = {}
        all_loss = 0
        all_auc = 0

        for idx, d in dataset.items():
            self._set_model_meta_parms(self._merge_weights(val_test_shared_weights, val_test_domain_weights[idx]))
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

    def separate_train_val_test(self, init_parms=True):
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

        # Save init weight
        weights = self.model.get_weights()
        for domain_idx, train_d in train_dataset.items():
            if not init_parms:
                # Reset adam learning rate
                opt = train.GradientDescentOptimizer(learning_rate=0.001)
                self.model.compile(loss=self.train_config['loss'], optimizer=opt,
                                   metrics=[AUC(num_thresholds=500, name="AUC")])
            self.model.set_weights(weights)
            self._set_model_meta_parms(
                self._merge_weights(self.best_shared_weights, self.best_domain_weights[domain_idx]))
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

    def _merge_weights(self, shared_weights, specific_weights):
        merged_weights = []
        if self.train_config['merged_method'] == 'plus':
            for x, y in zip(shared_weights, specific_weights):
                merged_weights.append(x + y)
        elif self.train_config['merged_method'] == 'times':
            for x, y in zip(shared_weights, specific_weights):
                merged_weights.append(x * y)
        return merged_weights

    def init_layer(self, model):
        session = K.get_session()
        for layer in model.layers:
            weights_initializer = tf.variables_initializer(layer.weights)
            session.run(weights_initializer)