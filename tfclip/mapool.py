from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="TFCLIP")
class MultiheadAttentionPooling(layers.Layer):
    def __init__(self, heads, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.heads = heads

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimensions of the inputs should be defined. "
                "Found `None`."
            )
        self.input_spec = InputSpec(ndim=3, axes={-1: channels})

        # noinspection PyAttributeOutsideInit
        self.query = self.add_weight(name="probe", shape=(1, 1, channels))

        # noinspection PyAttributeOutsideInit
        self.mhsa = layers.MultiHeadAttention(
            self.heads,
            channels // self.heads,
            name="mhsa",
            dtype=self.dtype_policy,
        )
        self.mhsa.build((input_shape[0], 1, channels), input_shape)

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        batch = ops.shape(inputs)[0]

        q = ops.repeat(self.query, batch, axis=0)
        x = self.mhsa(q, inputs)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[:1] + (1,) + input_shape[-1:]

    def get_config(self):
        config = super().get_config()
        config.update({"heads": self.heads})

        return config
