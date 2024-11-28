import warnings

from keras.src import initializers
from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="TFCLIP")
class ImagePositionEmbedding(layers.Layer):
    def __init__(self, patch_size, pretrain_size, cls_tok=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

        self.patch_size = patch_size
        self.pretrain_size = pretrain_size
        self.cls_tok = cls_tok

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )
        self.input_spec = InputSpec(ndim=3, axes={-1: channels})

        current_patches = (self.pretrain_size // self.patch_size) ** 2 + int(
            self.cls_tok
        )

        # noinspection PyAttributeOutsideInit
        self.embedding = self.add_weight(
            name="embedding",
            shape=[1, current_patches, channels],
            initializer=initializers.TruncatedNormal(stddev=0.02),
        )

        super().build(input_shape)

    def set_weights(self, weights):
        if self.built:
            if 1 != len(weights):
                raise ValueError(
                    f"You called `set_weights(weights)` on layer "
                    f'"{self.name}" with a weight list of length '
                    f"{len(weights)}, but the layer was expecting 1 weight. "
                    f"Provided weights: {weights}"
                )

            if 3 != len(weights[0].shape):
                raise ValueError(
                    f"Layer {self.name} weight shape {self.embedding.shape} "
                    f"is not compatible with provided "
                    f"weight shape {weights[0].shape}."
                )

            current_size = int(
                (weights[0].shape[1] - int(self.cls_tok)) ** 0.5
                * self.patch_size
            )
            if self.pretrain_size != current_size:
                warnings.warn(
                    f"Resizing absolute position embeddings from "
                    f"{current_size} to {self.pretrain_size}"
                )
                pretrain_patches = self.pretrain_size // self.patch_size
                current_patches = current_size // self.patch_size

                cls_embed, pos_embed = (
                    weights[0][:, : int(self.cls_tok)],
                    weights[0][:, int(self.cls_tok) :],
                )
                pos_embed = ops.reshape(
                    pos_embed, [1, current_patches, current_patches, -1]
                )
                pos_embed = ops.image.resize(
                    pos_embed,
                    [pretrain_patches, pretrain_patches],
                    interpolation="bicubic",
                )
                pos_embed = ops.reshape(pos_embed, [1, pretrain_patches**2, -1])

                weights = [ops.concatenate([cls_embed, pos_embed], axis=1)]

        super().set_weights(weights)

    def call(self, inputs, *args, **kwargs):
        outputs = inputs + self.embedding

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "pretrain_size": self.pretrain_size,
                "cls_tok": self.cls_tok,
            }
        )

        return config


@register_keras_serializable(package="TFCLIP")
class TextPositionEmbedding(layers.Layer):
    def __init__(self, context, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.supports_masking = True

        self.context = context

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )
        self.input_spec = InputSpec(ndim=3, axes={-1: channels})

        # noinspection PyAttributeOutsideInit
        self.embedding = self.add_weight(
            name="embedding",
            shape=[1, self.context, channels],
            initializer=initializers.TruncatedNormal(stddev=0.02),
        )

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        length = ops.shape(inputs)[1]
        length = ops.minimum(length, self.context)
        outputs = inputs[:, :length] + self.embedding[:, :length]

        return outputs

    def compute_output_shape(self, input_shape):
        if input_shape[1] is None:
            return input_shape

        return (
            input_shape[:1]
            + (min(input_shape[1], self.context),)
            + input_shape[2:]
        )

    def get_config(self):
        config = super().get_config()
        config.update({"context": self.context})

        return config
