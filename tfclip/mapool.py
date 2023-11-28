import tensorflow as tf
from keras import layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFCLIP')
class MultiheadAttentionPooling(layers.Layer):
    def __init__(self, heads, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)
        self.heads = heads

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=3, axes={-1: channels})

        # noinspection PyAttributeOutsideInit
        self.query = self.add_weight(name='probe', shape=(1, 1, channels))

        # noinspection PyAttributeOutsideInit
        self.mhsa = layers.MultiHeadAttention(self.heads, channels // self.heads, name='mhsa')

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        batch = tf.shape(inputs)[0]

        q = tf.repeat(self.query, batch, axis=0)
        x = self.mhsa(q, inputs)

        return x

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:1] + (1,) + input_shape[-1:]

    def get_config(self):
        config = super().get_config()
        config.update({'heads': self.heads})

        return config
