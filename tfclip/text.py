from keras.src import layers
from keras.src import models

from tfclip.abspos import TextPositionEmbedding
from tfclip.clstok import AddClassToken
from tfclip.lscale import LayerScale
from tfclip.qgelu import q_gelu
from tfclip.tlcm import TokenLastCausalMask
from tfclip.txtpool import TextGlobalPool


def TextTransformer(embed_dim, text_cfg, quick_gelu, custom=False, name="text"):
    head_width = text_cfg.width // text_cfg.heads

    ln_epsilon = 1.001e-5
    if text_cfg.norm_kwargs:
        norm_kwargs_ = set(text_cfg.norm_kwargs.keys()) - {"eps"}
        if set(text_cfg.norm_kwargs.keys()) - {"eps"}:
            raise ValueError(
                f"Unsupported normalization arguments in config: {norm_kwargs_}"
            )

        ln_epsilon = text_cfg.norm_kwargs.get("eps", ln_epsilon)

    if text_cfg.act_kwargs is not None:
        act_kwargs_ = set(text_cfg.act_kwargs.keys())
        raise ValueError(
            f"Unsupported activation arguments in config: {act_kwargs_}"
        )

    use_cls = custom and text_cfg.embed_cls
    mlp_act = q_gelu if quick_gelu else "gelu"

    # Define model inputs
    text = layers.Input(shape=(None,), dtype="int64", name=f"{name}_texts")

    # Define model pipeline
    x = text
    x = layers.Embedding(
        text_cfg.vocab_size, text_cfg.width, name=f"{name}_token_embed"
    )(x)

    if use_cls:
        x = AddClassToken(first=False, name=f"{name}_token_cls")(x)
    x = TextPositionEmbedding(
        text_cfg.context_length + int(use_cls), name=f"{name}_token_pos"
    )(x)

    if use_cls and not text_cfg.no_causal_mask:
        tlcm = TokenLastCausalMask(name=f"{name}_token_tlcm")(text)

    for i in range(text_cfg.layers):
        y = layers.LayerNormalization(
            epsilon=ln_epsilon, name=f"{name}_layer_{i}_attn_norm"
        )(x)
        if use_cls:
            y = layers.MultiHeadAttention(
                text_cfg.heads, head_width, name=f"{name}_layer_{i}_attn_mhsa"
            )(y, y, attention_mask=tlcm)
        else:
            y = layers.MultiHeadAttention(
                text_cfg.heads, head_width, name=f"{name}_layer_{i}_attn_mhsa"
            )(y, y, use_causal_mask=not text_cfg.no_causal_mask)
        if text_cfg.ls_init_value is not None:
            y = LayerScale(name=f"{name}_layer_{i}_attn_scale")(y)
        x = layers.add([x, y], name=f"{name}_layer_{i}_attn_add")

        y = layers.LayerNormalization(
            epsilon=ln_epsilon, name=f"{name}_layer_{i}_mlp_norm"
        )(x)
        y = layers.Dense(
            int(text_cfg.width * text_cfg.mlp_ratio),
            name=f"{name}_layer_{i}_mlp_expand",
        )(y)
        y = layers.Activation(mlp_act, name=f"{name}_layer_{i}_mlp_act")(y)
        y = layers.Dense(text_cfg.width, name=f"{name}_layer_{i}_mlp_squeeze")(
            y
        )
        if text_cfg.ls_init_value is not None:
            y = LayerScale(name=f"{name}_layer_{i}_mlp_scale")(y)
        x = layers.add([x, y], name=f"{name}_layer_{i}_mlp_add")

    x = layers.Activation("linear", name=f"{name}_head_in")(x)

    if use_cls:
        # presence of appended cls embed (CoCa) overrides pool_type,
        # always take cls token
        pooled, _ = TextGlobalPool("last", name=f"{name}_head_pool")([x, text])
        pooled = layers.LayerNormalization(
            epsilon=ln_epsilon, name=f"{name}_head_norm"
        )(pooled)
    else:
        x = layers.LayerNormalization(
            epsilon=ln_epsilon, name=f"{name}_head_norm"
        )(x)
        pooled, _ = TextGlobalPool(
            text_cfg.pool_type, name=f"{name}_head_pool"
        )([x, text])

    if "linear" == text_cfg.proj_type:
        pooled = layers.Dense(
            embed_dim, use_bias=text_cfg.proj_bias, name=f"{name}_head_proj"
        )(pooled)

    pooled = layers.Activation("linear", name=f"{name}_head_out")(pooled)

    model = models.Model(inputs=text, outputs=pooled, name=name)

    return model
