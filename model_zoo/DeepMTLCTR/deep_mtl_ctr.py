import os
import os.path as osp
import random
import time

import numpy as np
import tensorflow as tf
from deepctr import models
from deepctr.feature_column import SparseFeat
from tensorflow.python import train
from tensorflow.python.keras import callbacks, backend

from utils.auc import AUC  # This can be replaced by tf.keras.AUC when tf version >=1.12
from ..base_model import BaseModel


class DeepMTLCTR(BaseModel):
    def __init__(self, dataset, config):
        super(DeepMTLCTR, self).__init__(dataset, config)

    def build_model(self):
        fixlen_feature_columns = self.build_inputs()
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns
        if "shared_bottom" in self.model_config['name']:
            model = models.SharedBottom(dnn_feature_columns, bottom_dnn_hidden_units=self.model_config['hidden_dim'],
                                        tower_dnn_hidden_units=self.model_config['tower_hidden_dim'],
                                        dnn_dropout=self.model_config['dropout'],
                                        task_types=['binary' for _ in range(self.n_domain)],
                                        task_names=[f"domain_{n}" for n in range(self.n_domain)])
        elif "mmoe" in self.model_config['name']:
            model = models.MMOE(dnn_feature_columns, num_experts=self.model_config['num_experts'],
                                expert_dnn_hidden_units=self.model_config['hidden_dim'],
                                tower_dnn_hidden_units=self.model_config['tower_hidden_dim'],
                                dnn_dropout=self.model_config['dropout'],
                                gate_dnn_hidden_units=self.model_config['gate_dnn_hidden_units'],
                                task_types=['binary' for _ in range(self.n_domain)],
                                task_names=[f"domain_{n}" for n in range(self.n_domain)])
        elif "ple" in self.model_config['name']:
            model = models.PLE(dnn_feature_columns, shared_expert_num = self.model_config['shared_expert_num'],
                               specific_expert_num=self.model_config['specific_expert_num'],
                               num_levels= self.model_config['num_levels'],
                               tower_dnn_hidden_units=self.model_config['tower_hidden_dim'],
                               expert_dnn_hidden_units=self.model_config['hidden_dim'],
                               gate_dnn_hidden_units=self.model_config['gate_dnn_hidden_units'],
                               dnn_dropout=self.model_config['dropout'],
                               task_types=['binary' for _ in range(self.n_domain)],
                               task_names=[f"domain_{n}" for n in range(self.n_domain)])

        model.summary()
        # Optimization
        if self.train_config['optimizer'] == 'adam':
            opt = train.AdamOptimizer(learning_rate=self.train_config['learning_rate'])
        else:
            opt = self.train_config['optimizer']

        domain_model_dict = {}
        for domain_idx in range(self.n_domain):
            # domain_output = model.get_layer(f"domain_{domain_idx}")
            # inputs_list = [model.get_layer('uid'), model.get_layer('pid'), model.get_layer('domain')]
            domain_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.outputs[domain_idx])
            domain_model.compile(loss=self.train_config['loss'], optimizer=opt,
                                 metrics=[AUC(num_thresholds=500, name="AUC")])
            domain_model_dict[domain_idx] = domain_model
        self.domain_model_dict = domain_model_dict
        return model

    def train(self):
        backend.get_session().run(tf.global_variables_initializer())

        train_sequence = list(range(self.n_domain))
        lock = False
        for epoch in range(self.train_config['epoch']):
            print("Epoch: {}".format(epoch), "-" * 30)
            # Train
            random.shuffle(train_sequence)
            for idx in train_sequence:
                d = self.dataset.train_dataset[idx]
                print("Train on: Domain {}".format(idx))
                old_time = time.time()
                self.domain_model_dict[idx].fit(d['data'], steps_per_epoch=d['n_step'], verbose=2, callbacks=[],
                                                epochs=epoch + 1, initial_epoch=epoch)
                print("Training time: ", time.time() - old_time)
            # Val
            print("Val Result: ")
            avg_loss, avg_auc, domain_loss, domain_auc = self.val_and_test("val")
            # Early Stopping
            if self.early_stop_step(avg_auc):
                break
            # Test
            print("Test Result: ")
            test_avg_loss, test_avg_auc, test_domain_loss, test_domain_auc = self.val_and_test("test")

            # Lock the graph for better performance
            if not lock:
                graph = tf.get_default_graph()
                graph.finalize()
                lock = True

    def build_inputs(self):
        user_feat = self.build_emb("uid", "user_emb", self.n_uid, self.model_config['user_dim'],
                                   trainable=self.train_config['emb_trainable'])
        item_feat = self.build_emb("pid", "item_emb", self.n_pid, self.model_config['item_dim'],
                                   trainable=self.train_config['emb_trainable'])
        domain_feat = SparseFeat("domain", self.n_domain, self.model_config['domain_dim'], embedding_name="domain_emb")

        return [user_feat, item_feat, domain_feat]

    def build_emb(self, input_name, emb_name, n, dim, trainable=True):
        if self.train_config['load_pretrain_emb']:
            embedding_matrix = np.zeros((n, dim))
            emb_dict = getattr(self.dataset, emb_name)
            for key in sorted(emb_dict.keys()):
                embedding_vector = np.asarray(emb_dict[key].split(" "), dtype='float32')
                embedding_matrix[int(key)] = embedding_vector
            emb_layer = SparseFeat(input_name, n, dim,
                                   embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                   embedding_name=emb_name, trainable=trainable)
        else:
            emb_layer = SparseFeat(input_name, self.n_uid, self.model_config['user_dim'], embedding_name=emb_name)
        return emb_layer

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
            domain_model = tf.keras.models.Model(inputs=self.model.inputs,
                                                 outputs=self.model.outputs[domain_idx])
            if not init_parms:
                # Reset adam learning rate
                opt = train.GradientDescentOptimizer(learning_rate=self.train_config['learning_rate'])
                domain_model.compile(loss=self.train_config['loss'], optimizer=opt,
                                     metrics=[AUC(num_thresholds=500, name="AUC")])
            else:
                domain_model.compile(loss=self.train_config['loss'], optimizer=self.train_config['optimizer'],
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
            domain_model.fit(train_d['data'].repeat(), steps_per_epoch=train_d['n_step'], verbose=verbose,
                             callbacks=callback,
                             validation_data=val_dataset[domain_idx]['data'].repeat(),
                             validation_steps=val_dataset[domain_idx]['n_step'],
                             epochs=self.train_config['epoch'])
            # Test
            domain_model.load_weights(chk_path)
            p_loss, p_auc = domain_model.evaluate(
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
            p_loss, p_auc = self.domain_model_dict[idx].evaluate(d['data'], steps=d['n_step'],
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
