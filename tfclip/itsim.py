import numpy as np
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="TFCLIP")
class ImageTextSimilarity(layers.Layer):
    def __init__(self, scale_init, bias_init, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=2), InputSpec(ndim=2)]

        self.scale_init = scale_init
        self.bias_init = bias_init

    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.scale = self.add_weight(
            name="scale",
            shape=[],
            initializer=initializers.Constant(self.scale_init),
            constraint=MinConstraint(np.log(100.0, dtype=self.dtype)),
        )

        if self.bias_init:
            # noinspection PyAttributeOutsideInit
            self.bias = self.add_weight(
                name="bias",
                shape=[],
                initializer=initializers.Constant(self.bias_init),
            )

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        image, text = inputs

        image /= ops.norm(image, axis=-1, keepdims=True)
        text /= ops.norm(text, axis=-1, keepdims=True)

        outputs = ops.matmul(
            image * ops.exp(self.scale), ops.moveaxis(text, -1, -2)
        )
        if self.bias_init:
            outputs += self.bias

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[1][0]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "scale_init": self.scale_init,
                "bias_init": self.bias_init,
            }
        )

        return config


@register_keras_serializable(package="TFCLIP")
class MinConstraint(constraints.Constraint):
    def __init__(self, min_value):
        self.min_value = min_value

    def __call__(self, w):
        w = backend.convert_to_tensor(w)
        w = ops.minimum(w, self.min_value)

        return w

    def get_config(self):
        return {"min_value": self.min_value}
