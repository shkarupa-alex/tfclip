from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="TFCLIP")
class TokenLastCausalMask(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=2, dtype="int64")

    def call(self, inputs, *args, **kwargs):
        length = ops.shape(inputs)[1]
        length1 = length + 1

        ones_mask = ops.ones((1, length1, length1), dtype="int32")
        row_index = ops.cumsum(ones_mask, axis=-2)
        col_index = ops.cumsum(ones_mask, axis=-1)
        mask = ops.greater_equal(row_index, col_index)

        cls_mask = (inputs > 0)[:, None]
        cls_mask = ops.pad(
            cls_mask, [[0, 0], [length, 0], [1, 0]], constant_values=True
        )

        return mask & cls_mask

    def compute_output_shape(self, input_shape):
        if input_shape[1] is None:
            return input_shape[0], None, None

        return input_shape[0], input_shape[1] + 1, input_shape[1] + 1
