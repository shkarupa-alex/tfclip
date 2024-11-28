from keras.src import initializers
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="TFCLIP")
class LayerScale(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )
        self.input_spec = InputSpec(ndim=3, axes={-1: channels})

        # noinspection PyAttributeOutsideInit
        self.gamma = self.add_weight(
            name="gamma",
            shape=[1, 1, channels],
            initializer=initializers.Constant(1e-5),
        )

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        return inputs * self.gamma

    def compute_output_shape(self, input_shape):
        return input_shape
