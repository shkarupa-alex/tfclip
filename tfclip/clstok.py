from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="TFCLIP")
class AddClassToken(layers.Layer):
    def __init__(self, first=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.first = first

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimensions of the inputs should be defined. "
                "Found `None`."
            )
        self.input_spec = InputSpec(ndim=3, axes={-1: channels})

        # noinspection PyAttributeOutsideInit
        self.token = self.add_weight(
            name="token", shape=(1, 1, channels), initializer="zeros"
        )

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        batch_size = ops.shape(inputs)[0]
        cls_token = ops.repeat(self.token, batch_size, axis=0)

        outputs = [cls_token, inputs] if self.first else [inputs, cls_token]
        outputs = ops.concatenate(outputs, axis=1)

        return outputs

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None

        token_mask = ops.ones_like(mask[:, :1], dtype="bool")

        mask = [token_mask, mask] if self.first else [mask, token_mask]
        mask = ops.concatenate(mask, axis=1)

        return mask

    def compute_output_shape(self, input_shape):
        length = input_shape[1]
        if length is None:
            return input_shape

        return input_shape[:1] + (length + 1,) + input_shape[2:]

    def get_config(self):
        config = super().get_config()
        config.update({"first": self.first})

        return config


@register_keras_serializable(package="TFCLIP")
class SplitClassToken(layers.Layer):
    def __init__(self, first=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.first = first

    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError(
                "Channel dimensions of the inputs should be defined. "
                "Found `None`."
            )
        self.input_spec = InputSpec(ndim=3, axes={-1: self.channels})

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        if self.first:
            token, features = ops.split(inputs, [1], axis=1)
        else:
            length = ops.shape(inputs)[1]
            features, token = ops.split(inputs, [length - 1], axis=1)
        token = ops.reshape(token, [-1, self.channels])

        return token, features

    def compute_output_shape(self, input_shape):
        token_shape = input_shape[:1] + (self.channels,)
        features_shape = input_shape[:1] + (
            None if input_shape[1] is None else input_shape[1] - 1,
            self.channels,
        )

        return token_shape, features_shape

    def get_config(self):
        config = super().get_config()
        config.update({"first": self.first})

        return config
