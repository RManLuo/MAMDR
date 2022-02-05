import math
import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend, callbacks
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from .weighted_loss import WeightedLoss
from ..base_model import BaseModel
from utils import tool
from tensorflow.python import train
from utils.auc import AUC  # This can be replaced by tf.keras.AUC when tf version >=1.12
from tensorflow.python.keras.models import Model
from tensorflow.python.ops import nn


class UncertaintyWeight(object):
    '''
    Train any model in uncertainty_weight manner
    '''

    def __init__(self, base_model: BaseModel):
        '''
        :param base_model: any model inherited from the base_model
        '''
        self.base_model = base_model
        self.add_weighted_loss()

    def __getattr__(self, item):
        '''
        Delegate the base model
        :param item:
        :return:
        '''
        return getattr(self.base_model, item)

    def add_weighted_loss(self):
        y_true = layers.Input(shape=(1,), dtype=tf.float32, name='label')
        user_id, item_id, domain_idx = self.model.inputs
        y_pred = self.model.outputs[0]
        y_pred = WeightedLoss(n_domains=self.n_domain)([y_true, y_pred, domain_idx])
        model = Model(inputs=[user_id, item_id, domain_idx, y_true], outputs=y_pred)
        model.summary()
        # Optimization
        if self.train_config['optimizer'] == 'adam':
            opt = train.AdamOptimizer(learning_rate=self.train_config['learning_rate'])
        else:
            opt = self.train_config['optimizer']

        model.compile(loss=None,
                      optimizer=opt,
                      metrics=[AUC(num_thresholds=500, name="AUC")])
        model.metrics_names = []  # Must set to empty to remove bug
        self.model = model

    def train(self):
        backend.get_session().run(tf.global_variables_initializer())

        tensorboard_callback = callbacks.TensorBoard(log_dir=os.path.dirname(self.checkpoint_path),
                                                     histogram_freq=self.train_config['histogram_freq'],
                                                     write_grads=True)
        data_iter = self.build_data_iter()
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
                self.model.fit(d['data'], steps_per_epoch=d['n_step'], verbose=0, callbacks=[],
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

    def build_data_iter(self):
        data_iter = {}
        for idx, d in self.dataset.train_dataset.items():
            train_iter = d['data'].make_initializable_iterator()
            data_iter[idx] = {"train_iter": train_iter, "train_step": d['n_step']}
        return data_iter
