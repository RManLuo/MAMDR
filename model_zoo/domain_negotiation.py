import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend, callbacks
from tensorflow.python.keras import backend as K

from model_zoo import BaseModel
from .maml import MAML


class DomainNegotiation(MAML):
    def __init__(self, base_model: BaseModel):
        super(DomainNegotiation, self).__init__(base_model)

    def train(self):
        print("Start Domain Negotiation on model: {}".format(self.model_config['name']))

        tensorboard_callback = callbacks.TensorBoard(log_dir=os.path.dirname(self.checkpoint_path),
                                                     histogram_freq=self.train_config['histogram_freq'],
                                                     write_grads=True)
        tensorboard_callback.set_model(self.model)

        # Initialize for Meta-learning
        self._get_model_meta_parms()
        # Save Meta Weight
        meta_weights = self._get_meta_weights()

        backend.get_session().run(tf.global_variables_initializer())
        # Get meta data
        meta_data_split, meta_sequence = self.build_meta_data_split()
        # Train across the different domains
        tensorboard_callback.on_train_begin()
        lock = False
        for epoch in range(self.train_config['epoch']):
            print("Epoch: {}".format(epoch), "-" * 30)
            tensorboard_callback.on_epoch_begin(epoch)
            # Shuffle meta sequence
            if self.train_config['shuffle_sequence']:
                random.shuffle(meta_sequence)

            if self.train_config['target_domain'] >= 0:
                train_sequence = meta_sequence + [self.train_config['target_domain']]
            else:
                train_sequence = meta_sequence

            # Set the Meta Weights for each epoch
            self._set_model_meta_parms(meta_weights)

            # Inner Train
            for idx in train_sequence:
                d = meta_data_split[idx]
                # Reset AUC metrics
                for m in self.model.stateful_metric_functions:
                    m.reset_states()

                running_loss = 0
                running_auc = 0

                # Initialize data iterator
                train_iter = d['train_iter']
                K.get_session().run(train_iter.initializer)
                train_step = d['train_step']

                if self.train_config['meta_train_step'] > 0 and idx not in [self.train_config['target_domain']]:
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

                print(
                    "Train on: Domain {}, Loss: {}, AUC: {}, Step: {}, Time: {}".format(idx, running_loss / train_step,
                                                                                        running_auc,
                                                                                        train_step,
                                                                                        time.time() - old_time))

            # Apply grads
            self._update_meta_weight(meta_weights)
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

