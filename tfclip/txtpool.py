import tensorflow as tf
from keras import layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFCLIP')
class TextGlobalPool(layers.Layer):
    def __init__(self, mode, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)

        if mode not in {'first', 'last', 'argmax', 'none'}:
            raise ValueError(f'Unsupported pooling mode: {mode}')

        self.mode = mode

    def call(self, inputs, *args, mask=None, **kwargs):
        if 'first' == self.mode:
            pooled, tokens = inputs[:, 0], inputs[:, 1:]
        elif 'last' == self.mode:
            pooled, tokens = inputs[:, -1], inputs[:, :-1]
        elif 'argmax' == self.mode:
            if mask is None:
                raise ValueError(f'Pooling with `argmax` mode requires mask')
            indices = tf.cast(mask, 'int32')
            indices = tf.reduce_sum(indices, axis=list(range(1, mask.shape.rank))) - 1
            pooled = tf.gather(inputs, indices, batch_dims=1)
            tokens = inputs
        else:
            pooled = tokens = inputs

        return pooled, tokens

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        if 'none' == self.mode:
            return input_shape, input_shape

        pooled_shape = input_shape[:1] + input_shape[2:]
        if 'argmax' == self.mode:
            return pooled_shape, input_shape

        length = None if input_shape[1] is None else input_shape[1] - 1
        tokens_shape = input_shape[:1] + (length,) + input_shape[2:]

        return pooled_shape, tokens_shape

    def get_config(self):
        config = super().get_config()
        config.update({'mode': self.mode})

        return config
