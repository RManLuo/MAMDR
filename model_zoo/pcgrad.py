import math
import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend, callbacks
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers

from model_zoo import BaseModel
from utils import tool


class PCGrad(object):
    '''
    Train any model in PCGrad manner
    '''

    def __init__(self, base_model: BaseModel):
        '''
        :param base_model: any model inherited from the base_model
        '''
        self.base_model = base_model

    def __getattr__(self, item):
        '''
        Delegate the base model
        :param item:
        :return:
        '''
        return getattr(self.base_model, item)

    def train(self):
        '''
        Train the model using PCGrad
        :return:
        '''
        print("Start PCGrad training on model: {}".format(self.model_config['name']))

        tensorboard_callback = callbacks.TensorBoard(log_dir=os.path.dirname(self.checkpoint_path),
                                                     histogram_freq=self.train_config['histogram_freq'],
                                                     write_grads=True)
        tensorboard_callback.set_model(self.model)

        # Initialize for Meta-learning
        self._get_model_meta_parms()
        # Save Meta Weight
        meta_weights = self._get_meta_weights()
        meta_train = self._make_meta_train_function()

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

                running_loss = 0
                running_auc = 0

                # Initialize data iterator
                train_iter = d['train_iter']
                K.get_session().run(train_iter.initializer)
                train_step = d['train_step']

                if self.train_config['meta_train_step'] > 0:
                    train_step = min(train_step, self.train_config['meta_train_step'])

                # Clear grads
                K.get_session().run(self.clear_grads)
                old_time = time.time()
                for step_index in range(train_step):
                    x, y, _ = self.model._standardize_user_data(train_iter)
                    loss, auc = meta_train(x + y)  # Accumulate grads
                    running_loss += loss
                    running_auc = auc
                    # Tensorboard callback
                    batch_logs = {'batch': step_index, 'size': 1, "domain_{}_AUC".format(idx): auc,
                                  "domain_{}_loss".format(idx): loss}
                    tensorboard_callback.on_batch_end(step_index, batch_logs)

                print("Train on: Domain {}, Loss: {}, AUC: {}, Time: {}".format(idx, running_loss / train_step,
                                                                                running_auc,
                                                                                time.time() - old_time))
                current_grads = K.batch_get_value(self.accum_grads)
                final_grads = current_grads

                # Sample support Domains
                candidate_domains = train_sequence.copy()
                candidate_domains.remove(idx)
                aux_idxs = random.sample(candidate_domains, k=self.train_config['sample_num'])
                for aux_idx in aux_idxs:
                    print(f"Support Domain: {aux_idx}, Query Domain: {idx}")
                    aux_d = meta_data_split[aux_idx]
                    # Initialize data iterator
                    aux_train_iter = aux_d['train_iter']
                    K.get_session().run(aux_train_iter.initializer)
                    aux_train_step = aux_d['train_step']
                    # Clear grads
                    K.get_session().run(self.clear_grads)

                    for _ in range(aux_train_step):
                        x, y, _ = self.model._standardize_user_data(aux_train_iter)
                        outs = meta_train(x + y)  # Accumulate grads

                    aux_grads = K.batch_get_value(self.accum_grads)
                    self.PCGrad(final_grads, current_grads, aux_grads)

                # Apply grad
                self.set_accum_grads(final_grads)
                self._meta_train_step()

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

    def PCGrad(self, final_grads, current_grads, aux_grads):
        for var in range(len(final_grads)):
            grad_dot = np.sum(current_grads[var] * aux_grads[var], axis=-1)
            # If Conflict
            aux_grads[var][grad_dot > 0] -= np.expand_dims(
                grad_dot[grad_dot > 0] / np.linalg.norm(current_grads[var][grad_dot > 0], axis=-1), -1) * \
                                            current_grads[var][
                                                grad_dot > 0]
            final_grads[var] += aux_grads[var]

    def _get_model_meta_parms(self):
        '''
        Set model parameters for meta-learning.
        :return:
        '''
        if self.train_config['meta_parms'][0] == "all":
            self.model_meta_parms = self.model.trainable_weights
        elif self.train_config['meta_parms'][0] == "all_hidden":
            meta_parms = []
            for p in self.model.trainable_weights:
                if "emb" in p.name:
                    continue
                meta_parms.append(p)
            self.model_meta_parms = meta_parms
        else:
            meta_parms = []
            for name in self.train_config['meta_parms']:
                found = False
                for p in self.model.trainable_weights:
                    if name in p.name:
                        meta_parms.append(p)
                        found = True
                if not found:
                    raise ValueError("meta parms: {} not found in the model".format(name))
            self.model_meta_parms = meta_parms
        # Set variables update op for meta-weights.
        self.set_meta_parms_weight_op = tool.SetVarOp(self.model_meta_parms)

    def _set_model_meta_parms(self, meta_weights):
        '''
        Update meta parameters
        :param meta_weights: A list of numpy array.
        :return:
        '''
        self.set_meta_parms_weight_op(meta_weights)

    def _get_meta_weights(self):
        '''
        Get the weights of meta parameters
        :return: A list of numpy array.
        '''
        return K.batch_get_value(self.model_meta_parms)

    def _make_meta_train_function(self):
        '''
        Define the outer update function for meta learning.
        :return: A K.function used for accumulating the gradients
        '''
        self.meta_optimizer = tf.train.AdamOptimizer(learning_rate=self.train_config['meta_learning_rate'])
        self.accum_grads = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in self.model_meta_parms]
        self.clear_grads = [tf.assign(ag, tf.zeros_like(ag)) for ag in self.accum_grads]
        self.set_accum_grads = tool.SetVarOp(self.accum_grads)
        updates = []
        grads = K.gradients(self.model.total_loss, self.model_meta_parms)

        # Meta Parms update
        if self.train_config['average_meta_grad'] == "mean" and self.train_config['meta_train_step'] > 0:
            grads_var = [(ag / float(self.n_domain * self.train_config['meta_train_step']), v) for ag, v in
                         zip(self.accum_grads, self.model_meta_parms)]
        else:
            grads_var = list(zip(self.accum_grads, self.model_meta_parms))

        # Parms update op
        self.meta_parms_update_step = self.meta_optimizer.apply_gradients(grads_var)

        # Accumulate the gradients
        for g, ag in zip(grads, self.accum_grads):
            if self.train_config['average_meta_grad'] == "moving_mean":
                updates.append(K.moving_average_update(ag, g, 0.999))
            elif self.train_config['average_meta_grad'] == "drop":
                if len(ag.shape) > 1:
                    updates.append(K.update_add(ag, g))
                # Only support for dense gradients
                else:
                    new_g = layers.Dropout(0.2)(g)
                    updates.append(K.update_add(ag, new_g))
            else:
                updates.append(K.update_add(ag, g))

        return K.function((self.model._feed_inputs +
                           self.model._feed_targets),
                          [self.model.total_loss] + self.model.metrics_tensors,
                          updates=updates)

    def _meta_train_step(self):
        # Update meta parameters
        K.get_session().run(self.meta_parms_update_step)

        # Clear grads
        K.get_session().run(self.clear_grads)

        return self._get_meta_weights()

    def meta_finetune_val(self):
        '''
        Finetune a few step on the meta parameters to evaluate the performance
        :return:
        '''
        train_dataset = self.dataset.train_dataset
        val_dataset = self.dataset.val_dataset

        train_epoch = self.train_config['meta_finetune_step']

        domain_loss = {}
        domain_auc = {}
        all_loss = 0
        all_auc = 0
        # Save meta weight
        weights = self.model.get_weights()
        for domain_idx, train_d in train_dataset.items():
            self.model.set_weights(weights)
            # Fine tune
            print("Finetune on domain: {}".format(domain_idx))

            for epoch in range(train_epoch):
                # Finetune
                self.model.fit(train_d['data'], steps_per_epoch=train_d['n_step'], verbose=0,
                               epochs=epoch + 1, initial_epoch=epoch)

            p_loss, p_auc = self.model.evaluate(
                val_dataset[domain_idx]['data'], steps=val_dataset[domain_idx]['n_step'],
                verbose=0)

            domain_loss[domain_idx] = p_loss
            domain_auc[domain_idx] = p_auc
            all_loss += p_loss
            all_auc += p_auc

        # Restore meta weight
        self.model.set_weights(weights)
        avg_loss = all_loss / len(domain_loss)
        avg_auc = all_auc / len(domain_auc)
        print("Loss: ", domain_loss)
        print("AUC: ", domain_auc)
        print("Overall val Loss: {}, AUC: {}".format(avg_loss, avg_auc))
        return avg_loss, avg_auc, domain_loss, domain_auc

    def shuffle_and_batch(self, dataset):
        '''Shuffle the meta-train and meta-val set'''
        return dataset.shuffle(self.dataset.shuffle_buffer_size).batch(
            self.dataset.batch_size)

    def build_meta_data_split(self):
        meta_data_split = {}

        for idx, d in self.dataset.train_dataset.items():
            train_iter = d['data'].make_initializable_iterator()
            meta_data_split[idx] = {"train_iter": train_iter, "train_step": d['n_step']}
        return meta_data_split

    def val(self):
        if self.train_config['meta_finetune_step'] > 0:
            # Finetune a few steps to evaluate the performance
            # Val
            print("Val Result: ")
            val_avg_loss, val_avg_auc, val_domain_loss, val_domain_auc = self.meta_finetune_val()
        else:
            # Val
            print("Val Result: ")
            val_avg_loss, val_avg_auc, val_domain_loss, val_domain_auc = self.val_and_test("val")
        return val_avg_loss, val_avg_auc, val_domain_loss, val_domain_auc

    @staticmethod
    def check_same(A, B):
        same = False
        for x, y in zip(A, B):
            if np.array_equal(x, y):
                print("Same")
                same = True
        if not same:
            print("Not same")
