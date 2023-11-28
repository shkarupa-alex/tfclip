import numpy as np
import tensorflow as tf
from keras import initializers, layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFCLIP')
class ImageTextSimilarity(layers.Layer):
    def __init__(self, scale_init, bias_init, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=2), layers.InputSpec(ndim=2)]

        self.scale_init = scale_init
        self.bias_init = bias_init

    @shape_type_conversion
    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.scale = self.add_weight(
            'scale', shape=[],
            initializer=initializers.constant(self.scale_init),
            constraint=lambda s: tf.minimum(s, np.log(100., dtype=self.dtype)))

        if self.bias_init:
            # noinspection PyAttributeOutsideInit
            self.bias = self.add_weight(
                'bias', shape=[],
                initializer=initializers.constant(self.bias_init))

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        image, text = inputs

        image /= tf.norm(image, axis=-1, keepdims=True)
        text /= tf.norm(text, axis=-1, keepdims=True)

        outputs = tf.matmul(image * tf.exp(self.scale), text, transpose_b=True)
        if self.bias_init:
            outputs += self.bias

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[1][0]

    def get_config(self):
        config = super().get_config()
        config.update({
            'scale_init': self.scale_init,
            'bias_init': self.bias_init,
        })

        return config
