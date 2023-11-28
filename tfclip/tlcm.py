import tensorflow as tf
from keras import layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFCLIP')
class TokenLastCausalMask(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=2, dtype='int64')

    def call(self, inputs, *args, **kwargs):
        length = tf.shape(inputs)[1]
        length1 = length + 1

        mask = tf.linalg.band_part(tf.ones((1, length1, length1), tf.bool), -1, 0)

        cls_mask = (inputs > 0)[:, None]
        cls_mask = tf.pad(cls_mask, [[0, 0], [length, 0], [1, 0]], constant_values=True)

        return mask & cls_mask

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        if input_shape[1] is None:
            return input_shape[0], None, None

        return input_shape[0], input_shape[1] + 1, input_shape[1] + 1
