import tensorflow as tf
from tf_keras import layers
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFCLIP')
class AttentionalPooler(layers.Layer):
    def __init__(self, units, heads, queries, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)
        self.units = units
        self.heads = heads
        self.queries = queries
        self.epsilon = epsilon

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=3, axes={-1: channels})

        # noinspection PyAttributeOutsideInit
        self.query = self.add_weight(name='query', shape=(1, self.queries, self.units))

        # noinspection PyAttributeOutsideInit
        self.mhsa = layers.MultiHeadAttention(self.heads, self.units // self.heads, name='mhsa')

        # noinspection PyAttributeOutsideInit
        self.ln_q = layers.LayerNormalization(epsilon=self.epsilon, name='ln_q')

        # noinspection PyAttributeOutsideInit
        self.ln_k = layers.LayerNormalization(epsilon=self.epsilon, name='ln_k')

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        batch = tf.shape(inputs)[0]

        q = self.ln_q(self.query)
        q = tf.repeat(q, batch, axis=0)

        k = self.ln_k(inputs)
        x = self.mhsa(q, k)

        return x

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:1] + (self.queries, self.units)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'heads': self.heads,
            'queries': self.queries,
            'epsilon': self.epsilon
        })

        return config
