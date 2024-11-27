from keras.src import layers
from keras.src import ops
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="TFCLIP")
class EvaMultiHeadAttention(layers.MultiHeadAttention):
    def __init__(
        self,
        num_heads,
        key_dim,
        value_dim=None,
        dropout=0.0,
        use_bias=True,
        output_shape=None,
        attention_axes=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        norm_epsilon=None,
        rpe_pretrain=None,
        rpe_temperature=10000.0,
        **kwargs,
    ):
        super().__init__(
            num_heads,
            key_dim,
            value_dim=value_dim,
            dropout=dropout,
            use_bias=use_bias,
            output_shape=output_shape,
            attention_axes=attention_axes,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.norm_epsilon = norm_epsilon
        self.rpe_pretrain = rpe_pretrain
        self.rpe_temperature = rpe_temperature

    def build(self, query_shape, value_shape, key_shape=None):
        self._query_shape = query_shape

        super().build(query_shape, value_shape, key_shape)

    def _build_attention(self, rank):
        if self.norm_epsilon:
            self.output_norm = layers.LayerNormalization(
                epsilon=self.norm_epsilon, name="attention_norm"
            )
            self.output_norm.build(
                self._query_shape[:-1] + (self.num_heads * self.value_dim,)
            )

        self._build_rpe()

        super()._build_attention(rank)

    def _build_rpe(self):
        if self.rpe_pretrain is None:
            return

        size = int(self._query_shape[1] ** 0.5)
        num_bands = self._query_shape[2] // self._num_heads // 4

        bands = 1.0 / (
            self.rpe_temperature
            ** (
                ops.arange(0, num_bands, 1, dtype=self.compute_dtype)
                / num_bands
            )
        )

        t = ops.arange(size, dtype=self.compute_dtype) * (
            self.rpe_pretrain / size
        )
        grid = ops.stack(ops.meshgrid(t, t, indexing="ij"), axis=-1)
        pos = grid[..., None] * bands

        pos_sin, pos_cos = ops.sin(pos), ops.cos(pos)

        pos_sin = ops.reshape(pos_sin, [size**2, 2 * num_bands])
        pos_cos = ops.reshape(pos_cos, [size**2, 2 * num_bands])

        self.rpe_sin = ops.repeat(pos_sin, [2] * 2 * num_bands, axis=-1)[
            None, :, None
        ]
        self.rpe_cos = ops.repeat(pos_cos, [2] * 2 * num_bands, axis=-1)[
            None, :, None
        ]

    def _apply_rpe(self, inputs):
        if self.rpe_pretrain is None:
            return inputs

        clstok = self._query_shape[1] - int(self._query_shape[1] ** 0.5) ** 2
        if clstok:
            token, inputs = ops.split(inputs, [1], axis=1)

        rotated = ops.stack([-inputs[..., 1::2], inputs[..., ::2]], axis=-1)
        rotated = ops.reshape(rotated, ops.shape(inputs))
        outputs = inputs * self.rpe_cos + rotated * self.rpe_sin

        if clstok:
            outputs = ops.concatenate([token, outputs], axis=1)

        return outputs

    def _compute_attention(
        self, query, key, value, attention_mask=None, training=None
    ):
        if self.rpe_pretrain:
            query = self._apply_rpe(query)
            key = self._apply_rpe(key)

        attention_output, attention_scores = super()._compute_attention(
            query, key, value, attention_mask=attention_mask, training=training
        )

        if self.norm_epsilon:
            dynamic_shape = ops.shape(attention_output)
            static_shape = attention_output.shape

            attention_output = ops.reshape(
                attention_output,
                dynamic_shape[:-2] + (static_shape[-1] * static_shape[-2],),
            )
            attention_output = self.output_norm(attention_output)
            attention_output = ops.reshape(attention_output, dynamic_shape)
            attention_output.set_shape(static_shape)

        return attention_output, attention_scores

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "norm_epsilon": self.norm_epsilon,
                "rpe_pretrain": self.rpe_pretrain,
                "rpe_temperature": self.rpe_temperature,
            }
        )

        return config
