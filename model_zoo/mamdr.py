import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks, backend

from model_zoo import BaseModel
from .specific_base_model import SpecificBase


class MAMDR(SpecificBase):
    def __init__(self, base_model: BaseModel):
        super(MAMDR, self).__init__(base_model)

    def train(self):
        print("Start MAMDR on model: {}".format(self.model_config['name']))

        tensorboard_callback = callbacks.TensorBoard(log_dir=os.path.dirname(self.checkpoint_path),
                                                     histogram_freq=self.train_config['histogram_freq'],
                                                     write_grads=True)
        tensorboard_callback.set_model(self.model)

        # Initialize for Meta-learning
        self._get_model_meta_parms()
        # Save Meta Weight
        self.meta_weights = self._get_meta_weights()
        self.domain_weights = {}
        for domain_idx in range(self.n_domain):
            self.init_layer(self.model)
            self.domain_weights[domain_idx] = self._get_meta_weights()

        backend.get_session().run(tf.global_variables_initializer())
        # Get meta data
        meta_data_split, train_sequence = self.build_meta_data_split()
        # Train across the different domains
        tensorboard_callback.on_train_begin()
        lock = False
        for epoch in range(self.train_config['epoch']):
            print("Epoch: {}".format(epoch), "-" * 30)
            tensorboard_callback.on_epoch_begin(epoch)
            # Shuffle meta sequence
            if self.train_config['shuffle_sequence']:
                random.shuffle(train_sequence)

            # Update Shared
            self._set_model_meta_parms(self.meta_weights)
            for idx in train_sequence:
                d = meta_data_split[idx]
                train_iter = d['train_iter']
                K.get_session().run(train_iter.initializer)
                self.model.fit(train_iter, steps_per_epoch=d['train_step'], verbose=0, epochs=epoch + 1,
                               initial_epoch=epoch)
            # Apply grads
            self._update_meta_weight(self.meta_weights, meta_lr=self.train_config['meta_learning_rate'])

            # Update specific
            for idx in train_sequence:
                old_time = time.time()
                d = meta_data_split[idx]
                # Initialize data iterator

                # Sample support Domains
                candidate_domains = train_sequence.copy()
                candidate_domains.remove(idx)
                aux_idxs = random.sample(candidate_domains, k=self.train_config['sample_num'])
                if self.train_config['add_query_domain']:
                    aux_idxs.append(idx)

                merged_weights = self._merge_weights(self.meta_weights, self.domain_weights[idx])
                accum_grads = [np.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in self.model_meta_parms]

                for aux_idx in aux_idxs:
                    print(f"Support Domain: {aux_idx}, Query Domain: {idx}")
                    # Set the Meta Weights
                    self._set_model_meta_parms(merged_weights)

                    aux_d = meta_data_split[aux_idx]
                    # Initialize data iterator
                    aux_train_iter = aux_d['train_iter']
                    K.get_session().run(aux_train_iter.initializer)
                    aux_train_step = aux_d['train_step']
                    for step_index in range(aux_train_step):
                        loss, auc = self.model.train_on_batch(aux_train_iter)

                    # Regularize on target domain
                    # Initialize data iterator
                    train_iter = d['train_iter']
                    K.get_session().run(train_iter.initializer)
                    train_step = d['train_step']
                    if self.train_config['domain_regulation_step'] > 0:
                        train_step = min(train_step, self.train_config['domain_regulation_step'])

                    for step_index in range(train_step):
                        loss, auc = self.model.train_on_batch(train_iter)

                    # Apply grads
                    if "batch" in self.model_config['name']:
                        self._accumulate_grad(accum_grads, merged_weights, self.meta_weights)
                    else:
                        self._update_meta_weight(self.domain_weights[idx], merged_weights,
                                                 meta_lr=self.train_config['meta_learning_rate'])
                        merged_weights = self._merge_weights(self.meta_weights, self.domain_weights[idx])

                if "batch" in self.model_config['name']:
                    self._update_meta_weight_by_grads(accum_grads, self.domain_weights[idx])

                # Finetune on target
                if self.train_config['finetune_every_epoch']:
                    # Initialize data iterator
                    train_iter = d['train_iter']
                    K.get_session().run(train_iter.initializer)
                    train_step = d['train_step']
                    merged_weights = self._merge_weights(self.meta_weights, self.domain_weights[idx])
                    self._set_model_meta_parms(merged_weights)

                    # Reset AUC metrics
                    for m in self.model.stateful_metric_functions:
                        m.reset_states()

                    running_loss = 0
                    running_auc = 0

                    for step_index in range(train_step):
                        loss, auc = self.model.train_on_batch(train_iter)
                        running_loss += loss
                        running_auc = auc
                        # Tensorboard callback
                        batch_logs = {'batch': step_index, 'size': 1, "domain_{}_AUC".format(idx): auc,
                                      "domain_{}_loss".format(idx): loss}
                        tensorboard_callback.on_batch_end(step_index, batch_logs)

                    print(
                        "Train on: Domain {}, Loss: {}, AUC: {}, Step: {}, Time: {}".format(idx,
                                                                                            running_loss / train_step,
                                                                                            running_auc,
                                                                                            train_step,
                                                                                            time.time() - old_time))

                    # Update specific only
                    self._update_domain_weights(self.domain_weights[idx], merged_weights)

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

    def _update_domain_weights(self, domain_weights, merged_weights):
        new_vars = self._get_meta_weights()
        for var in range(len(new_vars)):
            domain_weights[var] = new_vars[var] - merged_weights[var]

    def _update_meta_weight(self, update_vars, merged_weights=None, meta_lr=1):
        new_vars = self._get_meta_weights()
        if merged_weights != None:
            old_vars = merged_weights
        else:
            old_vars = update_vars
        for var in range(len(new_vars)):
            update_vars[var] += (new_vars[var] - old_vars[var]) * meta_lr

    def _accumulate_grad(self, accum_grads, old_vars, shared_weights, train_step=1, normalize_grads=True):
        new_vars = self._get_meta_weights()
        if not normalize_grads:
            train_step = 1
        for var in range(len(accum_grads)):
            if self.train_config['merged_method'] == 'plus':
                accum_grads[var] += (new_vars[var] - old_vars[var]) / train_step  # Normalize gradient
            elif self.train_config['merged_method'] == 'times':
                accum_grads[var] += (new_vars[var] - old_vars[var]) * shared_weights[
                    var] / train_step  # Normalize gradient

    def _update_meta_weight_by_grads(self, grads, old_vars):
        for var in range(len(old_vars)):
            old_vars[var] += grads[var] / self.train_config['sample_num'] * self.train_config['meta_learning_rate']
            grads[var] = np.zeros_like(grads[var])  # Clear grads
