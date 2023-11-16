import numpy as np
from keras import backend, layers, models
from keras.src.applications import imagenet_utils
from tfclip.abspos import ImagePositionEmbedding
from tfclip.attnpool import AttentionalPooler
from tfclip.clstok import AddClassToken, SplitClassToken
from tfclip.lscale import LayerScale
from tfclip.qgelu import q_gelu


def VisionTransformer(embed_dim, vision_cfg, quick_gelu, img_mean, img_std, name='vision'):
    num_heads = vision_cfg.width // vision_cfg.head_width

    ln_epsilon = 1.001e-5
    if vision_cfg.norm_kwargs:
        norm_kwargs_ = set(vision_cfg.norm_kwargs.keys()) - {'eps'}
        if set(vision_cfg.norm_kwargs.keys()) - {'eps'}:
            raise ValueError(f'Unsupported normalization arguments in config: {norm_kwargs_}')

        ln_epsilon = vision_cfg.norm_kwargs.get('eps', ln_epsilon)

    if vision_cfg.act_kwargs is not None:
        act_kwargs_ = set(vision_cfg.act_kwargs.keys())
        raise ValueError(f'Unsupported activation arguments in config: {act_kwargs_}')

    if vision_cfg.patch_dropout:
        raise ValueError(f'Unsupported patch dropout in config: {vision_cfg.patch_dropout}')

    if not isinstance(vision_cfg.attentional_pool, bool):
        raise ValueError(f'Unsupported attentional pooling mode in config: {vision_cfg.attentional_pool}')

    if 'learnable' != vision_cfg.pos_embed_type:
        raise ValueError(f'Unsupported positional embedding type in config: {vision_cfg.pos_embed_type}')

    mlp_act = q_gelu if quick_gelu else 'gelu'

    # Define model inputs
    input_shape = imagenet_utils.obtain_input_shape(
        None, default_size=vision_cfg.image_size, min_size=vision_cfg.patch_size, data_format='channel_last',
        require_flatten=False)
    image = layers.Input(shape=input_shape, dtype='uint8', name=f'{name}/images')

    # Define model pipeline
    x = image

    x = layers.Normalization(
        mean=np.array(img_mean) * 255., variance=(np.array(img_std) * 255.) ** 2, name=f'{name}/image/norm')(x)
    x = layers.Conv2D(
        vision_cfg.width, vision_cfg.patch_size, strides=vision_cfg.patch_size, use_bias=False,
        name=f'{name}/patch/embed')(x)
    x = layers.Reshape(
        [(vision_cfg.image_size // vision_cfg.patch_size) ** 2, vision_cfg.width], name=f'{name}/patch/flatten')(x)
    x = AddClassToken(name=f'{name}/patch/cls')(x)
    x = ImagePositionEmbedding(vision_cfg.patch_size, vision_cfg.image_size, name=f'{name}/patch/pos')(x)
    if not vision_cfg.no_ln_pre:
        x = layers.LayerNormalization(epsilon=ln_epsilon, name=f'{name}/patch/norm')(x)

    for i in range(vision_cfg.layers):
        y = layers.LayerNormalization(epsilon=ln_epsilon, name=f'{name}/layer_{i}/attn/norm')(x)
        y = layers.MultiHeadAttention(
            num_heads, vision_cfg.width // num_heads, name=f'{name}/layer_{i}/attn/mhsa')(y, y)
        if vision_cfg.ls_init_value is not None:
            y = LayerScale(name=f'{name}/layer_{i}/attn/scale')(y)
        x = layers.add([x, y], name=f'{name}/layer_{i}/attn/add')

        y = layers.LayerNormalization(epsilon=ln_epsilon, name=f'{name}/layer_{i}/mlp/norm')(x)
        y = layers.Dense(int(vision_cfg.width * vision_cfg.mlp_ratio), name=f'{name}/layer_{i}/mlp/expand')(y)
        y = layers.Activation(mlp_act, name=f'{name}/layer_{i}/mlp/act')(y)
        y = layers.Dense(vision_cfg.width, name=f'{name}/layer_{i}/mlp/squeeze')(y)
        if vision_cfg.ls_init_value is not None:
            y = LayerScale(name=f'{name}/layer_{i}/mlp/scale')(y)
        x = layers.add([x, y], name=f'{name}/layer_{i}/mlp/add')

    if vision_cfg.attentional_pool:
        x = AttentionalPooler(
            embed_dim, vision_cfg.attn_pooler_heads, vision_cfg.attn_pooler_queries, epsilon=ln_epsilon,
            name=f'{name}/head/attn')(x)
        x = layers.LayerNormalization(epsilon=ln_epsilon, name=f'{name}/head/norm')(x)
        pooled, tokens = GlobalPool(
            vision_cfg.pool_type, vision_cfg.patch_size, vision_cfg.image_size, name=f'{name}/head/pool')(x)
    elif vision_cfg.final_ln_after_pool:
        pooled, tokens = GlobalPool(
            vision_cfg.pool_type, vision_cfg.patch_size, vision_cfg.image_size, name=f'{name}/head/pool')(x)
        pooled = layers.LayerNormalization(epsilon=ln_epsilon, name=f'{name}/head/norm')(pooled)
    else:
        x = layers.LayerNormalization(epsilon=ln_epsilon, name=f'{name}/head/norm')(x)
        pooled, tokens = GlobalPool(
            vision_cfg.pool_type, vision_cfg.patch_size, vision_cfg.image_size, name=f'{name}/head/pool')(x)

    pooled = layers.Dense(embed_dim, use_bias=False, name=f'{name}/head/proj')(pooled)

    outputs = [pooled, tokens] if vision_cfg.output_tokens else pooled
    model = models.Model(inputs=image, outputs=outputs, name=name)

    return model


def GlobalPool(pool_type, patch_size, image_size, name=None):
    if name is None:
        counter = backend.get_uid('global_pool')
        name = f'global_pool_{counter}'

    def apply(inputs):
        if 'avg' == pool_type:
            _, tokens = SplitClassToken(patch_size, image_size, name=f'{name}/split')(inputs)
            pooled = layers.GlobalAvgPool2D(name=f'{name}/avg')(tokens)
        elif 'tok' == pool_type:
            pooled, tokens = SplitClassToken(patch_size, image_size, name=f'{name}/split')(inputs)
        else:
            pooled = tokens = inputs

        return pooled, tokens

    return apply
