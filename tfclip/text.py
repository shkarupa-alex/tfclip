from keras import layers, models
from tfclip.abspos import TextPositionEmbedding
from tfclip.clstok import AddClassToken
from tfclip.lscale import LayerScale
from tfclip.qgelu import q_gelu
from tfclip.tlcm import TokenLastCausalMask
from tfclip.txtpool import TextGlobalPool


def TextTransformer(embed_dim, text_cfg, quick_gelu, custom=False, name='text'):
    head_width = text_cfg.width // text_cfg.heads

    ln_epsilon = 1.001e-5
    if text_cfg.norm_kwargs:
        norm_kwargs_ = set(text_cfg.norm_kwargs.keys()) - {'eps'}
        if set(text_cfg.norm_kwargs.keys()) - {'eps'}:
            raise ValueError(f'Unsupported normalization arguments in config: {norm_kwargs_}')

        ln_epsilon = text_cfg.norm_kwargs.get('eps', ln_epsilon)

    if text_cfg.act_kwargs is not None:
        act_kwargs_ = set(text_cfg.act_kwargs.keys())
        raise ValueError(f'Unsupported activation arguments in config: {act_kwargs_}')

    use_cls = custom and text_cfg.embed_cls
    mlp_act = q_gelu if quick_gelu else 'gelu'

    # Define model inputs
    text = layers.Input(shape=(None,), dtype='int64', name=f'{name}/texts')

    # Define model pipeline
    x = text
    x = layers.Embedding(text_cfg.vocab_size, text_cfg.width, name=f'{name}/token/embed')(x)

    if use_cls:
        x = AddClassToken(first=False, name=f'{name}/token/cls')(x)
    x = TextPositionEmbedding(text_cfg.context_length + int(use_cls), name=f'{name}/token/pos')(x)

    if use_cls and not text_cfg.no_causal_mask:
        tlcm = TokenLastCausalMask(name=f'{name}/token/tlcm')(text)

    for i in range(text_cfg.layers):
        y = layers.LayerNormalization(epsilon=ln_epsilon, name=f'{name}/layer_{i}/attn/norm')(x)
        if use_cls:
            y = layers.MultiHeadAttention(text_cfg.heads, head_width, name=f'{name}/layer_{i}/attn/mhsa')(
                y, y, attention_mask=tlcm)
        else:
            y = layers.MultiHeadAttention(text_cfg.heads, head_width, name=f'{name}/layer_{i}/attn/mhsa')(
                y, y, use_causal_mask=not text_cfg.no_causal_mask)
        if text_cfg.ls_init_value is not None:
            y = LayerScale(name=f'{name}/layer_{i}/attn/scale')(y)
        x = layers.add([x, y], name=f'{name}/layer_{i}/attn/add')

        y = layers.LayerNormalization(epsilon=ln_epsilon, name=f'{name}/layer_{i}/mlp/norm')(x)
        y = layers.Dense(int(text_cfg.width * text_cfg.mlp_ratio), name=f'{name}/layer_{i}/mlp/expand')(y)
        y = layers.Activation(mlp_act, name=f'{name}/layer_{i}/mlp/act')(y)
        y = layers.Dense(text_cfg.width, name=f'{name}/layer_{i}/mlp/squeeze')(y)
        if text_cfg.ls_init_value is not None:
            y = LayerScale(name=f'{name}/layer_{i}/mlp/scale')(y)
        x = layers.add([x, y], name=f'{name}/layer_{i}/mlp/add')

    x = layers.Activation('linear', name=f'{name}/head/in')(x)

    if use_cls:
        # presence of appended cls embed (CoCa) overrides pool_type, always take cls token
        pooled, _ = TextGlobalPool('last', name=f'{name}/head/pool')([x, text])
        pooled = layers.LayerNormalization(epsilon=ln_epsilon, name=f'{name}/head/norm')(pooled)
    else:
        x = layers.LayerNormalization(epsilon=ln_epsilon, name=f'{name}/head/norm')(x)
        pooled, _ = TextGlobalPool(text_cfg.pool_type, name=f'{name}/head/pool')([x, text])
    pooled = layers.Dense(embed_dim, use_bias=text_cfg.proj_bias, name=f'{name}/head/proj')(pooled)

    pooled = layers.Activation('linear', name=f'{name}/head/out')(pooled)

    model = models.Model(inputs=text, outputs=pooled, name=name)

    return model
