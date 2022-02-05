from tensorflow.python.keras.layers import Input, Dense, Lambda, Layer
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.python.ops import nn

# 自定义loss层
class WeightedLoss(Layer):
    def __init__(self, n_domains, **kwargs):
        self.nb_outputs = n_domains
        self.is_placeholder = True
        super(WeightedLoss, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # 初始化 log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        self.log_vars = self.add_weight(
            "log_var",
            shape=[self.nb_outputs, 1],
            initializer=Constant(1.),
            dtype="float32",
            trainable=True)
        super(WeightedLoss, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred, domain_idx):
        var = nn.embedding_lookup(self.log_vars, domain_idx)
        weights = tf.div(1., var ** 2)
        log_var = tf.log(var)
        loss = K.mean(weights * K.binary_crossentropy(ys_true, ys_pred) + log_var, axis=-1)
        return loss

    def call(self, inputs, **kwargs):
        ys_true, ys_pred, domain_idx = inputs
        # Partition by domain
        idx = tf.cast(domain_idx[0, 0], tf.int32)
        loss = self.multi_loss(ys_true, ys_pred, idx)
        self.add_loss(loss, inputs=inputs)
        return ys_pred
