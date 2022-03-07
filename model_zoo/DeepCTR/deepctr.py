import random
import time

import deepctr.layers as layers
import numpy as np
import tensorflow as tf
from deepctr import models
from deepctr.feature_column import SparseFeat, build_input_features, input_from_feature_columns
from tensorflow.python import train
from tensorflow.python.keras import backend

from utils.auc import AUC  # This can be replaced by tf.keras.AUC when tf version >=1.12
from ..base_model import BaseModel


class DeepCTR(BaseModel):
    def __init__(self, dataset, config):
        super(DeepCTR, self).__init__(dataset, config)

    def build_model(self):
        fixlen_feature_columns = self.build_inputs()
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns
        if 'mlp' in self.model_config['name']:
            model = self.build_mlp(dnn_feature_columns,
                                   dnn_hidden_units=self.model_config['hidden_dim'],
                                   dnn_dropout=self.model_config['dropout'])
        elif "wdl" in self.model_config['name']:
            model = models.WDL(linear_feature_columns, dnn_feature_columns,
                               dnn_hidden_units=self.model_config['hidden_dim'],
                               dnn_dropout=self.model_config['dropout'])
        elif "nfm" in self.model_config['name']:
            model = models.NFM(linear_feature_columns, dnn_feature_columns,
                               dnn_hidden_units=self.model_config['hidden_dim'],
                               dnn_dropout=self.model_config['dropout'])
        elif "autoint" in self.model_config['name']:
            model = models.AutoInt(linear_feature_columns, dnn_feature_columns,
                                   dnn_hidden_units=self.model_config['hidden_dim'], att_head_num=4,
                                   dnn_dropout=self.model_config['dropout'])
        elif "ccpm" in self.model_config['name']:
            model = models.CCPM(linear_feature_columns, dnn_feature_columns,
                                dnn_hidden_units=self.model_config['hidden_dim'],
                                dnn_dropout=self.model_config['dropout'])
        elif "pnn" in self.model_config['name']:
            model = models.PNN(dnn_feature_columns, dnn_hidden_units=self.model_config['hidden_dim'],
                               dnn_dropout=self.model_config['dropout'])
        elif "deepfm" in self.model_config['name']:
            model = models.DeepFM(linear_feature_columns, dnn_feature_columns,
                                  dnn_hidden_units=self.model_config['hidden_dim'],
                                  dnn_dropout=self.model_config['dropout'])

        model.summary()
        # Optimization
        if self.train_config['optimizer'] == 'adam':
            opt = train.AdamOptimizer(learning_rate=self.train_config['learning_rate'])
        else:
            opt = self.train_config['optimizer']

        model.compile(loss=self.train_config['loss'], optimizer=opt,
                      metrics=[AUC(num_thresholds=500, name="AUC")])
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
                self.model.fit(d['data'], steps_per_epoch=d['n_step'], verbose=2, callbacks=[],
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
            emb_layer = SparseFeat(input_name, n, dim, embedding_name=emb_name)
        return emb_layer

    def build_mlp(self, dnn_feature_columns, dnn_hidden_units=(256, 128, 64), l2_reg_linear=0.00001,
                  l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0., dnn_activation='relu',
                  task='binary'):
        features = build_input_features(dnn_feature_columns)

        inputs_list = list(features.values())

        sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                             l2_reg_embedding, seed)

        dnn_input = layers.utils.combined_dnn_input(sparse_embedding_list, dense_value_list)
        dnn_out = layers.core.DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(
            dnn_input)
        dnn_logit = tf.keras.layers.Dense(
            1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_out)
        output = layers.core.PredictionLayer(task)(dnn_logit)

        model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
        return model
