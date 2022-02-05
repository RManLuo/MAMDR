import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend, callbacks
from tensorflow.python.keras import backend as K

from model_zoo import BaseModel
from .maml import MAML


class Reptile(MAML):
    def __init__(self, base_model: BaseModel):
        super(Reptile, self).__init__(base_model)

    def train(self):
        print("Start reptile on model: {}".format(self.model_config['name']))

        tensorboard_callback = callbacks.TensorBoard(log_dir=os.path.dirname(self.checkpoint_path),
                                                     histogram_freq=self.train_config['histogram_freq'],
                                                     write_grads=True)
        tensorboard_callback.set_model(self.model)

        # Initialize for Meta-learning
        self._get_model_meta_parms()
        # Save Meta Weight
        meta_weights = self._get_meta_weights()
        # Accumulate grads for batch version
        accum_grads = [np.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in self.model_meta_parms]

        backend.get_session().run(tf.global_variables_initializer())
        # Get meta data
        meta_data_split = self.build_meta_data_split()
        train_sequence = list(range(self.n_domain))
        # Train across the different domains
        tensorboard_callback.on_train_begin()
        lock = False
        for epoch in range(self.train_config['epoch']):
            print("Epoch: {}".format(epoch), "-" * 30)
            tensorboard_callback.on_epoch_begin(epoch)

            # Inner Train
            random.shuffle(train_sequence)
            for idx in train_sequence:
                if self.train_config['target_domain'] >= 0 and idx in [self.train_config['target_domain']]:
                    continue
                d = meta_data_split[idx]
                # Reset AUC metrics
                for m in self.model.stateful_metric_functions:
                    m.reset_states()

                # Set the Meta Weights for each task
                self._set_model_meta_parms(meta_weights)

                running_loss = 0
                running_auc = 0

                # Initialize data iterator
                train_iter = d['train_iter']
                K.get_session().run(train_iter.initializer)
                train_step = d['train_step']

                if self.train_config['meta_train_step'] > 0:
                    train_step = min(train_step, self.train_config['meta_train_step'])

                old_time = time.time()
                for step_index in range(train_step):
                    loss, auc = self.model.train_on_batch(train_iter)
                    running_loss += loss
                    running_auc = auc
                    # Tensorboard callback
                    batch_logs = {'batch': step_index, 'size': 1, "domain_{}_AUC".format(idx): auc,
                                  "domain_{}_loss".format(idx): loss}
                    tensorboard_callback.on_batch_end(step_index, batch_logs)

                print("Train on: Domain {}, Loss: {}, AUC: {}, Time: {}".format(idx, running_loss / train_step,
                                                                                running_auc,
                                                                                time.time() - old_time))

                if self.train_config['target_domain'] >= 0:
                    K.get_session().run(self.target_iter.initializer)
                    self.model.fit(self.target_iter, steps_per_epoch=1, verbose=0,
                                   epochs=epoch + 1, initial_epoch=epoch)

                #  Accumulate grade or update
                if "batch" in self.model_config['name']:
                    self._accumulate_grad(accum_grads, meta_weights)
                else:
                    self._update_meta_weight(meta_weights)

            # Apply grads
            if "batch" in self.model_config['name']:
                self._update_meta_weight_by_grads(accum_grads, meta_weights)

            self._set_model_meta_parms(meta_weights)
            if self.train_config['target_domain'] >= 0:
                K.get_session().run(self.target_iter.initializer)
                print(f"Train on target domain: {self.train_config['target_domain']}")
                self.model.fit(self.target_iter, steps_per_epoch=self.target_step, verbose=2, callbacks=[],
                               epochs=epoch + 1, initial_epoch=epoch)

            if epoch % self.train_config['val_every_step'] == 0:
                val_avg_loss, val_avg_auc, val_domain_loss, val_domain_auc = self.val()
                epoch_logs = {"val_avg_loss": np.array(val_avg_loss), "val_avg_auc": np.array(val_avg_auc)}
                for k, v in val_domain_loss.items():
                    epoch_logs["val_domain_{}_loss".format(k)] = np.array(v)
                    epoch_logs["val_domain_{}_auc".format(k)] = np.array(val_domain_auc[k])
                tensorboard_callback.on_epoch_end(epoch, epoch_logs)
                # Early Stopping
                val_metric = val_domain_auc[self.train_config['target_domain']] if self.train_config[
                                                                                       'target_domain'] >= 0 else val_avg_auc
                if self.early_stop_step(val_metric):
                    break
                # Test
                print("Test Result: ")
                test_avg_loss, test_avg_auc, test_domain_loss, test_domain_auc = self.val_and_test("test")

            # Lock the graph for better performance
            if not lock:
                graph = tf.get_default_graph()
                graph.finalize()
                lock = True
        tensorboard_callback.on_train_end()

    def _update_meta_weight(self, old_vars):
        new_vars = self._get_meta_weights()
        for var in range(len(new_vars)):
            old_vars[var] += (
                    (new_vars[var] - old_vars[var]) * self.train_config['meta_learning_rate']
            )

    def _accumulate_grad(self, accum_grads, old_vars):
        new_vars = self._get_meta_weights()
        for var in range(len(accum_grads)):
            accum_grads[var] += new_vars[var] - old_vars[var]

    def _update_meta_weight_by_grads(self, grads, old_vars):
        for var in range(len(old_vars)):
            old_vars[var] += grads[var] * self.train_config['meta_learning_rate']
            grads[var] = np.zeros_like(grads[var])  # Clear grads

    def build_meta_data_split(self):
        meta_data_split = {}
        if self.train_config['target_domain'] >= 0:
            target_domain = self.train_config['target_domain']
            target_data = self.dataset.train_dataset[target_domain]
            self.target_iter = target_data['data'].make_initializable_iterator()
            self.target_step = target_data['n_step']

        for idx, d in self.dataset.train_dataset.items():
            train_iter = d['data'].make_initializable_iterator()
            meta_data_split[idx] = {"train_iter": train_iter, "train_step": d['n_step']}
        return meta_data_split
