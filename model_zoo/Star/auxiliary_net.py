import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import InputSpec
from tensorflow.python.keras.layers import Layer
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops


class AuxiliaryNet(Layer):

    def __init__(self,
                 n_domain,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(AuxiliaryNet, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        self.n_domain = n_domain
        self.units = int(units)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = False

    def build(self, input_shape):
        input_shape, domain_indicator_shape = input_shape
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = [InputSpec(min_ndim=2,
                                     axes={-1: input_shape[-1].value}),
                           InputSpec(shape=domain_indicator_shape)]

        # Multi Domain
        self.PN_Kernel = self.add_weight(
                    "kernel_specific",
                    shape=[self.n_domain, input_shape[-1].value, self.units],
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                    dtype=self.dtype,
                    trainable=True)
        if self.use_bias:
            self.PN_Bias = self.add_weight(
                "bias_specific",
                shape=[self.n_domain, self.units, ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.PN_Bias.append(None)

        self.built = True

    def _domain_partition(self, idx):
        p_kernel = nn.embedding_lookup(self.PN_Kernel, idx)
        p_bias = nn.embedding_lookup(self.PN_Bias, idx)

        self.kernel = p_kernel
        if self.use_bias:
            self.bias = p_bias
        else:
            self.bias = None

    def call(self, inputs, **kwargs):
        # Unpack to original inputs and domain_indicator
        inputs, domain_indicator = inputs
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        rank = common_shapes.rank(inputs)

        # Partition by domain
        idx = tf.cast(domain_indicator[0, 0], tf.int32)
        self._domain_partition(idx)

        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = gen_math_ops.mat_mul(inputs, self.kernel)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = {
            'n_domain': self.n_domain,
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(AuxiliaryNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
