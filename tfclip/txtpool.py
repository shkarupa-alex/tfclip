import tensorflow as tf
from keras import layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFCLIP')
class TextGlobalPool(layers.Layer):
    def __init__(self, mode, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=3), layers.InputSpec(ndim=2, dtype='int64')]

        if mode not in {'first', 'last', 'argmax', 'none'}:
            raise ValueError(f'Unsupported pooling mode: {mode}')

        self.mode = mode

    def call(self, inputs, *args, **kwargs):
        tokens, texts = inputs

        if 'first' == self.mode:
            pooled, tokens = tokens[:, 0], tokens[:, 1:]
        elif 'last' == self.mode:
            pooled, tokens = tokens[:, -1], tokens[:, :-1]
        elif 'argmax' == self.mode:
            lendiff = tf.shape(tokens)[1] - tf.shape(texts)[1]
            indices = tf.cast(texts > 0, 'int32')
            indices = tf.reduce_sum(indices, axis=1) + lendiff - 1
            pooled = tf.gather(tokens, indices, batch_dims=1)
            tokens = tokens
        else:
            pooled = tokens = tokens

        return pooled, tokens

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        if 'none' == self.mode:
            return input_shape[0], input_shape[0]

        pooled_shape = input_shape[0][:1] + input_shape[0][2:]
        if 'argmax' == self.mode:
            return pooled_shape, input_shape[0]

        length = None if input_shape[0][1] is None else input_shape[0][1] - 1
        tokens_shape = input_shape[0][:1] + (length,) + input_shape[0][2:]

        return pooled_shape, tokens_shape

    def get_config(self):
        config = super().get_config()
        config.update({'mode': self.mode})

        return config
