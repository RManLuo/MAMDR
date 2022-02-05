import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.python import train
from tensorflow.python.keras import layers, backend, callbacks
from tensorflow.python.keras.models import Model

from utils.auc import AUC  # This can be replaced by tf.keras.AUC when tf version >=1.12
from .partitioned_norm import PartitionedNorm
from .star_fcn import StarFCN
from .auxiliary_net import AuxiliaryNet
from ..base_model import BaseModel


class Star(BaseModel):
    def __init__(self, dataset, config):
        super(Star, self).__init__(dataset, config)

    def build_model(self):
        model = self.build_model_structure()
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

        tensorboard_callback = callbacks.TensorBoard(log_dir=os.path.dirname(self.checkpoint_path),
                                                     histogram_freq=self.train_config['histogram_freq'],
                                                     write_grads=True)
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

    def build_model_structure(self):
        inputs, x = self.build_inputs()
        domain_indicator = inputs[-1]

        # PartitionedNorm
        if self.model_config['norm'] == "pn":
            x = PartitionedNorm(self.n_domain)([x, domain_indicator])
        elif self.model_config['norm'] == "bn":
            x = layers.BatchNormalization()(x)

        # Auxiliary Network
        aux_out = AuxiliaryNet(self.n_domain, self.model_config['auxiliary_dim'], activation="relu")(
            [x, domain_indicator])

        # Star FCN
        if self.model_config['dense'] == "dense":
            for h_dim in self.model_config['hidden_dim']:
                x = layers.Dense(h_dim, activation="relu")(x)
        elif self.model_config['dense'] == "star":
            for h_dim in self.model_config['hidden_dim']:
                x = StarFCN(self.n_domain, h_dim, activation="relu")([x, domain_indicator])

        if self.model_config['auxiliary_net']:
            x = layers.Add()([x, aux_out])

        output = layers.Dense(1, activation="sigmoid")(x)
        return Model(inputs=inputs, outputs=output)

    def build_inputs(self):
        user_input_layer = layers.Input(shape=(1,), dtype=tf.int32, name='uid')
        item_input_layer = layers.Input(shape=(1,), dtype=tf.int32, name='pid')
        domain_input_layer = layers.Input(shape=(1,), dtype=tf.int32, name='domain')

        user_emb = self.build_emb("user_emb", self.n_uid, self.model_config['user_dim'],
                                  trainable=self.train_config['emb_trainable'])(user_input_layer)  # B * 1 * d
        item_emb = self.build_emb("item_emb", self.n_pid, self.model_config['item_dim'],
                                  trainable=self.train_config['emb_trainable'])(item_input_layer)
        domain_emb = layers.Embedding(self.n_domain, self.model_config['domain_dim'], name="domain_emb")(
            domain_input_layer)

        input_feat = layers.Concatenate(-1)([user_emb, item_emb, domain_emb])  # B * 1 * 3d
        input_feat = layers.Flatten()(input_feat)  # B * 3d

        return [user_input_layer, item_input_layer, domain_input_layer], input_feat

    def build_emb(self, attr_name, n, dim, trainable=True):
        if self.train_config['load_pretrain_emb']:
            embedding_matrix = np.zeros((n, dim))
            emb_dict = getattr(self.dataset, attr_name)
            for key in sorted(emb_dict.keys()):
                embedding_vector = np.asarray(emb_dict[key].split(" "), dtype='float32')
                embedding_matrix[int(key)] = embedding_vector
            emb_layer = layers.Embedding(n, dim, name=attr_name,
                                         embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                         trainable=trainable)
        else:
            emb_layer = layers.Embedding(n, dim, name=attr_name)
        return emb_layer
