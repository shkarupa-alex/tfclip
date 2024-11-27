import numpy as np
from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src.applications import imagenet_utils

from tfclip.abspos import ImagePositionEmbedding
from tfclip.attnpool import AttentionalPooler
from tfclip.clstok import AddClassToken
from tfclip.clstok import SplitClassToken
from tfclip.evattn import EvaMultiHeadAttention
from tfclip.lscale import LayerScale
from tfclip.mapool import MultiheadAttentionPooling
from tfclip.qgelu import q_gelu


def VisionTransformer(
    embed_dim, vision_cfg, quick_gelu, img_mean, img_std, name="vision"
):
    num_heads = vision_cfg.width // vision_cfg.head_width

    ln_epsilon = 1.001e-5
    if vision_cfg.norm_kwargs:
        norm_kwargs_ = set(vision_cfg.norm_kwargs.keys()) - {"eps"}
        if set(vision_cfg.norm_kwargs.keys()) - {"eps"}:
            raise ValueError(
                f"Unsupported normalization arguments in config: {norm_kwargs_}"
            )

        ln_epsilon = vision_cfg.norm_kwargs.get("eps", ln_epsilon)

    if vision_cfg.act_kwargs is not None:
        act_kwargs_ = set(vision_cfg.act_kwargs.keys())
        raise ValueError(
            f"Unsupported activation arguments in config: {act_kwargs_}"
        )

    if vision_cfg.patch_dropout:
        raise ValueError(
            f"Unsupported patch dropout in config: {vision_cfg.patch_dropout}"
        )

    if not isinstance(vision_cfg.attentional_pool, bool):
        raise ValueError(
            f"Unsupported attentional pooling mode "
            f"in config: {vision_cfg.attentional_pool}"
        )

    if "learnable" != vision_cfg.pos_embed_type:
        raise ValueError(
            f"Unsupported positional embedding type "
            f"in config: {vision_cfg.pos_embed_type}"
        )

    if vision_cfg.swi_glu:
        mlp_act = "silu"
    elif quick_gelu:
        mlp_act = q_gelu
    else:
        mlp_act = "gelu"

    # Define model inputs
    input_shape = imagenet_utils.obtain_input_shape(
        None,
        default_size=vision_cfg.image_size,
        min_size=vision_cfg.patch_size,
        data_format="channel_last",
        require_flatten=False,
    )
    image = layers.Input(
        shape=input_shape, dtype="uint8", name=f"{name}_images"
    )

    # Define model pipeline
    x = image

    x = layers.Normalization(
        mean=np.array(img_mean) * 255.0,
        variance=(np.array(img_std) * 255.0) ** 2,
        name=f"{name}_image_norm",
    )(x)
    x = layers.Conv2D(
        vision_cfg.width,
        vision_cfg.patch_size,
        strides=vision_cfg.patch_size,
        use_bias=vision_cfg.patch_bias,
        name=f"{name}_patch_embed",
    )(x)
    x = layers.Reshape(
        [
            (vision_cfg.image_size // vision_cfg.patch_size) ** 2,
            vision_cfg.width,
        ],
        name=f"{name}_patch_flatten",
    )(x)
    if vision_cfg.embed_cls:
        x = AddClassToken(name=f"{name}_patch_cls")(x)
    x = ImagePositionEmbedding(
        vision_cfg.patch_size,
        vision_cfg.image_size,
        cls_tok=vision_cfg.embed_cls,
        name=f"{name}_patch_pos",
    )(x)
    if not vision_cfg.no_ln_pre:
        x = layers.LayerNormalization(
            epsilon=ln_epsilon, name=f"{name}_patch_norm"
        )(x)

    for i in range(vision_cfg.layers):
        if vision_cfg.post_norm:
            y = x
        else:
            y = layers.LayerNormalization(
                epsilon=ln_epsilon, name=f"{name}_layer_{i}_attn_norm"
            )(x)
        if vision_cfg.attn_norm or vision_cfg.rpe_pretrain:
            y = EvaMultiHeadAttention(
                num_heads,
                vision_cfg.width // num_heads,
                norm_epsilon=ln_epsilon,
                rpe_pretrain=vision_cfg.rpe_pretrain,
                name=f"{name}_layer_{i}_attn_mhsa",
            )(y, y)
        else:
            y = layers.MultiHeadAttention(
                num_heads,
                vision_cfg.width // num_heads,
                name=f"{name}_layer_{i}_attn_mhsa",
            )(y, y)
        if vision_cfg.post_norm:
            y = layers.LayerNormalization(
                epsilon=ln_epsilon, name=f"{name}_layer_{i}_attn_norm"
            )(y)
        if vision_cfg.ls_init_value is not None:
            y = LayerScale(name=f"{name}_layer_{i}_attn_scale")(y)
        x = layers.add([x, y], name=f"{name}_layer_{i}_attn_add")

        if vision_cfg.post_norm:
            y = x
        else:
            y = layers.LayerNormalization(
                epsilon=ln_epsilon, name=f"{name}_layer_{i}_mlp_norm"
            )(x)
        if vision_cfg.swi_glu:
            g = layers.Dense(
                int(vision_cfg.width * vision_cfg.mlp_ratio),
                name=f"{name}_layer_{i}_mlp_gate",
            )(y)
            g = layers.Activation(mlp_act, name=f"{name}_layer_{i}_mlp_act")(g)
            y = layers.Dense(
                int(vision_cfg.width * vision_cfg.mlp_ratio),
                name=f"{name}_layer_{i}_mlp_expand",
            )(y)
            y = layers.multiply([y, g], name=f"{name}_layer_{i}_mlp_multiply")
            y = layers.LayerNormalization(
                epsilon=ln_epsilon, name=f"{name}_layer_{i}_mlp_normalize"
            )(y)
            y = layers.Dense(
                vision_cfg.width, name=f"{name}_layer_{i}_mlp_squeeze"
            )(y)
        else:
            y = layers.Dense(
                int(vision_cfg.width * vision_cfg.mlp_ratio),
                name=f"{name}_layer_{i}_mlp_expand",
            )(y)
            y = layers.Activation(mlp_act, name=f"{name}_layer_{i}_mlp_act")(y)
            y = layers.Dense(
                vision_cfg.width, name=f"{name}_layer_{i}_mlp_squeeze"
            )(y)
        if vision_cfg.post_norm:
            y = layers.LayerNormalization(
                epsilon=ln_epsilon, name=f"{name}_layer_{i}_mlp_norm"
            )(y)
        if vision_cfg.ls_init_value is not None:
            y = LayerScale(name=f"{name}_layer_{i}_mlp_scale")(y)
        x = layers.add([x, y], name=f"{name}_layer_{i}_mlp_add")

    x = layers.Activation("linear", name=f"{name}_head_in")(x)

    if vision_cfg.attentional_pool:
        x = AttentionalPooler(
            embed_dim,
            vision_cfg.attn_pooler_heads,
            vision_cfg.attn_pooler_queries,
            epsilon=ln_epsilon,
            name=f"{name}_head_attnpool",
        )(x)
        x = layers.LayerNormalization(
            epsilon=ln_epsilon, name=f"{name}_head_norm"
        )(x)
        pooled = GlobalPool(vision_cfg.pool_type, name=f"{name}_head_pool")(x)
    elif vision_cfg.ma_pool:
        pooled = layers.LayerNormalization(
            epsilon=ln_epsilon, name=f"{name}_head_norm"
        )(x)
        pooled = MultiheadAttentionPooling(
            num_heads, name=f"{name}_head_mapool"
        )(pooled)
        y = layers.LayerNormalization(
            epsilon=ln_epsilon, name=f"{name}_head_mapool_mlp_norm"
        )(pooled)
        y = layers.Dense(
            int(vision_cfg.width * vision_cfg.mlp_ratio),
            name=f"{name}_head_mapool_mlp_expand",
        )(y)
        y = layers.Activation(mlp_act, name=f"{name}_head_mapool_mlp_act")(y)
        y = layers.Dense(
            vision_cfg.width, name=f"{name}_head_mapool_mlp_squeeze"
        )(y)
        pooled = layers.add([pooled, y], name=f"{name}_head_mapool_mlp_add")
        pooled = GlobalPool("first", name=f"{name}_head_pool")(pooled)
    elif vision_cfg.final_ln_after_pool:
        pooled = GlobalPool(vision_cfg.pool_type, name=f"{name}_head_pool")(x)
        pooled = layers.LayerNormalization(
            epsilon=ln_epsilon, name=f"{name}_head_norm"
        )(pooled)
    else:
        x = layers.LayerNormalization(
            epsilon=ln_epsilon, name=f"{name}_head_norm"
        )(x)
        pooled = GlobalPool(vision_cfg.pool_type, name=f"{name}_head_pool")(x)

    if not vision_cfg.ma_pool:
        pooled = layers.Dense(
            embed_dim, use_bias=vision_cfg.proj_bias, name=f"{name}_head_proj"
        )(pooled)

    pooled = layers.Activation("linear", name=f"{name}_head_out")(pooled)

    model = models.Model(inputs=image, outputs=pooled, name=name)

    return model


def GlobalPool(pool_type, name=None):
    if name is None:
        counter = backend.get_uid("global_pool")
        name = f"global_pool_{counter}"

    def apply(inputs):
        if "avg" == pool_type:
            _, tokens = SplitClassToken(name=f"{name}_split")(inputs)
            pooled = layers.GlobalAveragePooling1D(name=f"{name}_avg")(tokens)
        elif "tok" == pool_type:
            pooled, _ = SplitClassToken(name=f"{name}_split")(inputs)
        elif "first" == pool_type:
            pooled, _ = SplitClassToken(name=f"{name}_split")(inputs)
        else:
            pooled = inputs

        return pooled

    return apply
