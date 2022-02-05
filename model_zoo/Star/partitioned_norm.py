import tensorflow as tf
from tensorflow.python import nn
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import InputSpec
from tensorflow.python.keras.layers import Layer
from tensorflow.python.ops import variables as tf_variables


class PartitionedNorm(Layer):
    def __init__(self,
                 n_domain,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(PartitionedNorm, self).__init__(**kwargs)
        self.n_domain = n_domain
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.supports_masking = False

    def build(self, input_shape):
        input_shape, domain_indicator_shape = input_shape
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = [InputSpec(ndim=len(input_shape),
                                     axes={self.axis: dim}),
                           InputSpec(shape=domain_indicator_shape)]
        shape = (self.n_domain, dim)

        # Partitioned Part
        self.PN_Gamma = self.add_weight(shape=shape,
                                        name="gamma_specific",
                                        initializer=self.gamma_initializer,
                                        regularizer=self.gamma_regularizer,
                                        constraint=self.gamma_constraint)
        self.PN_Beta = self.add_weight(shape=shape,
                                       name="beta_specific",
                                       initializer=self.beta_initializer,
                                       regularizer=self.beta_regularizer,
                                       constraint=self.beta_constraint)
        self.PN_Mean = []
        self.PN_Var = []
        for idx in range(self.n_domain):
            self.PN_Mean.append(self.add_weight(
                shape=(dim,),
                name="moving_mean_{}".format(idx),
                initializer=self.moving_mean_initializer,
                synchronization=tf_variables.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf_variables.VariableAggregation.MEAN))

            self.PN_Var.append(self.add_weight(
                shape=(dim,),
                name="moving_variance_".format(idx),
                initializer=self.moving_variance_initializer,
                synchronization=tf_variables.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf_variables.VariableAggregation.MEAN))

        # Shared Part
        self.Shared_Gamma = self.add_weight(shape=(dim,),
                                            name='gamma_shared',
                                            initializer=self.gamma_initializer,
                                            regularizer=self.gamma_regularizer,
                                            constraint=self.gamma_constraint)
        self.Shared_Beta = self.add_weight(shape=(dim,),
                                           name='beta_shared',
                                           initializer=self.beta_initializer,
                                           regularizer=self.beta_regularizer,
                                           constraint=self.beta_constraint)

        self.built = True

    def _domain_partition(self, idx):
        pn_gamma = nn.embedding_lookup(self.PN_Gamma, idx)
        pn_beta = nn.embedding_lookup(self.PN_Beta, idx)

        self.gamma = tf.multiply(self.Shared_Gamma, pn_gamma)
        self.beta = tf.add(self.Shared_Beta, pn_beta)

        # self.moving_mean = nn.embedding_lookup(self.PN_Mean, idx, partition_strategy='div')
        # self.moving_variance = nn.embedding_lookup(self.PN_Var, idx, partition_strategy='div')

        mean_paris = []
        var_paris = []

        for i in range(self.n_domain):
            mean_paris.append((tf.equal(idx, i), (lambda k: lambda: self.PN_Mean[k])(i)))
            var_paris.append((tf.equal(idx, i), (lambda k: lambda: self.PN_Var[k])(i)))

        self.moving_mean = tf.case(mean_paris, exclusive=False, name="mean_update_switch")
        self.moving_variance = tf.case(var_paris, exclusive=False, name="var_update_switch")

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        # Unpack to original inputs and domain_indicator
        inputs, domain_indicator = inputs
        # Partition by domain
        idx = tf.cast(domain_indicator[0, 0], tf.int32)
        self._domain_partition(idx)

        input_shape = inputs.get_shape().as_list()
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explictly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)

                broadcast_beta = K.reshape(self.beta, broadcast_shape)

                broadcast_gamma = K.reshape(self.gamma,
                                            broadcast_shape)

                return K.batch_normalization(
                    inputs,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    epsilon=self.epsilon)
            else:
                return K.batch_normalization(
                    inputs,
                    self.moving_mean,
                    self.moving_variance,
                    self.beta,
                    self.gamma,
                    epsilon=self.epsilon)

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return normalize_inference()

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = K.normalize_batch_in_training(
            inputs, self.gamma, self.beta, reduction_axes,
            epsilon=self.epsilon)

        # Update running mean and var by domain
        mean_update_paris = []
        var_update_paris = []

        for i in range(self.n_domain):
            mean_update_paris.append(
                (tf.equal(idx, i), (lambda k: lambda: K.moving_average_update(self.PN_Mean[k], mean,
                                                                              self.momentum))(i)))
            var_update_paris.append((tf.equal(idx, i), (
                lambda k: lambda: K.moving_average_update(self.PN_Var[k], variance,
                                                          self.momentum))(i)))

        mean_update_switch = tf.case(mean_update_paris, exclusive=False, name="mean_update_switch")
        var_update_switch = tf.case(var_update_paris, exclusive=False, name="var_update_switch")

        self.add_update([mean_update_switch,
                         var_update_switch],
                        inputs=True)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def get_config(self):
        config = {
            'n_domain': self.n_domain,
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(PartitionedNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
