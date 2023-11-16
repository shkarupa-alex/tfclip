from keras import initializers, layers
from keras.saving import register_keras_serializable
from keras.src.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFCLIP')
class LayerScale(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=3, axes={-1: channels})

        # noinspection PyAttributeOutsideInit
        self.gamma = self.add_weight(
            'gamma', shape=[1, 1, channels],
            initializer=initializers.constant(1e-5))

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        return inputs * self.gamma

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
