import numpy as np
from dataclasses import dataclass
from keras import layers, models
from typing import Optional, Tuple, Union
from tfclip.itsim import ImageTextSimilarity
from tfclip.text import TextTransformer
from tfclip.vision import VisionTransformer


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    patch_bias: bool = False
    image_size: Union[Tuple[int, int], int] = 224

    embed_cls: bool = True
    ls_init_value: Optional[float] = None
    patch_dropout: float = 0.
    attentional_pool: bool = False
    attn_pooler_queries: int = 256
    attn_pooler_heads: int = 8
    ma_pool: bool = False
    no_ln_pre: bool = False
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False
    pool_type: str = 'tok'
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None

    # TensorFlow specific vision tower config
    attn_norm: Optional[bool] = None
    rpe_pretrain: Optional[int] = None
    post_norm: bool = False
    swi_glu: bool = False
    proj_bias: bool = False


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    tokenizer_kwargs: Optional[dict] = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'argmax'
    proj_bias: bool = False
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None

    # TensorFlow specific text tower config
    sp_tokenizer_name: Optional[str] = None
    hub_tokenizer_name: Optional[str] = None


def _build_vision_tower(embed_dim, vision_cfg, quick_gelu, img_mean, img_std):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    model = VisionTransformer(embed_dim, vision_cfg, quick_gelu, img_mean, img_std)

    return model


def _build_text_tower(embed_dim, text_cfg, quick_gelu, custom=False):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    model = TextTransformer(embed_dim, text_cfg, quick_gelu, custom=custom)

    return model


def CLIP(
        embed_dim, vision_cfg, text_cfg, img_mean, img_std, quick_gelu=False, scale_init=-np.log(0.07),
        bias_init=None, custom_text=False):
    vision = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, img_mean, img_std)
    text = _build_text_tower(embed_dim, text_cfg, quick_gelu, custom=custom_text)

    s = ImageTextSimilarity(scale_init, bias_init, name='head/sim')([vision.outputs[0], text.outputs[0]])
    s = layers.Activation('softmax', name='head/prob', dtype='float32')(s)

    model = models.Model(inputs=[vision.inputs[0], text.inputs[0]], outputs=s, name='clip')

    return model
