from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="TFCLIP")
class AttentionalPooler(layers.Layer):
    def __init__(self, units, heads, queries, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.units = units
        self.heads = heads
        self.queries = queries
        self.epsilon = epsilon

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimensions of the inputs should be defined. "
                "Found `None`."
            )
        self.input_spec = InputSpec(ndim=3, axes={-1: channels})

        # noinspection PyAttributeOutsideInit
        self.query = self.add_weight(
            name="query", shape=(1, self.queries, self.units)
        )

        # noinspection PyAttributeOutsideInit
        self.mhsa = layers.MultiHeadAttention(
            self.heads,
            self.units // self.heads,
            name="mhsa",
            dtype=self.dtype_policy,
        )
        self.mhsa.build((input_shape[0], self.queries, self.units), input_shape)

        # noinspection PyAttributeOutsideInit
        self.ln_q = layers.LayerNormalization(
            epsilon=self.epsilon, name="ln_q", dtype=self.dtype_policy
        )
        self.ln_q.build(self.query.shape)

        # noinspection PyAttributeOutsideInit
        self.ln_k = layers.LayerNormalization(
            epsilon=self.epsilon, name="ln_k", dtype=self.dtype_policy
        )
        self.ln_k.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        batch = ops.shape(inputs)[0]

        q = self.ln_q(self.query)
        q = ops.repeat(q, batch, axis=0)

        k = self.ln_k(inputs)
        x = self.mhsa(q, k)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[:1] + (self.queries, self.units)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "heads": self.heads,
                "queries": self.queries,
                "epsilon": self.epsilon,
            }
        )

        return config
