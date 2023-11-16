import tensorflow as tf


def q_gelu(x, name=None):
    with tf.name_scope(name or 'QuickGELU'):
        return x * tf.nn.sigmoid(1.702 * x)
