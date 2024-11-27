from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="TFCLIP")
class TextGlobalPool(layers.Layer):
    def __init__(self, mode, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=2, dtype="int64")]

        if mode not in {"first", "last", "argmax", "none"}:
            raise ValueError(f"Unsupported pooling mode: {mode}")

        self.mode = mode

    def call(self, inputs, *args, **kwargs):
        tokens, texts = inputs

        if "first" == self.mode:
            pooled, tokens = tokens[:, 0], tokens[:, 1:]
        elif "last" == self.mode:
            pooled, tokens = tokens[:, -1], tokens[:, :-1]
        elif "argmax" == self.mode:
            length = ops.shape(tokens)[1]
            lendiff = length - ops.shape(texts)[1]
            indices = ops.cast(texts > 0, "int32")
            indices = ops.sum(indices, axis=1) + lendiff - 1
            mask = ops.arange(length)[None] == indices[:, None]
            pooled = tokens[mask]
            tokens = tokens
        else:
            pooled = tokens = tokens

        return pooled, tokens

    def compute_output_shape(self, input_shape):
        if "none" == self.mode:
            return input_shape[0], input_shape[0]

        pooled_shape = input_shape[0][:1] + input_shape[0][2:]
        if "argmax" == self.mode:
            return pooled_shape, input_shape[0]

        length = None if input_shape[0][1] is None else input_shape[0][1] - 1
        tokens_shape = input_shape[0][:1] + (length,) + input_shape[0][2:]

        return pooled_shape, tokens_shape

    def get_config(self):
        config = super().get_config()
        config.update({"mode": self.mode})

        return config
