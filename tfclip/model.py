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

    timm_model_name: Optional[str] = None
    timm_model_pretrained: bool = False
    timm_pool: str = 'avg'
    timm_proj: str = 'linear'
    timm_proj_bias: bool = False
    timm_drop: float = 0.
    timm_drop_path: Optional[float] = None


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
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

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = 'mlp'
    hf_pooler_type: str = 'mean_pooler'  # attentional pooling for HF models

    # TensorFlow specific text tower config
    sp_tokenizer_name: Optional[str] = None
    hub_tokenizer_name: Optional[str] = None


def _build_vision_tower(embed_dim, vision_cfg, quick_gelu, img_mean, img_std):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    if vision_cfg.timm_model_name:
        raise ValueError(f'Unsupported image encoder: {vision_cfg.timm_model_name}')
    elif isinstance(vision_cfg.layers, (tuple, list)):
        raise ValueError(f'Unsupported image encoder: modified_resnet')
    else:
        model = VisionTransformer(embed_dim, vision_cfg, quick_gelu, img_mean, img_std)

    return model


def _build_text_tower(embed_dim, text_cfg, quick_gelu, custom=False):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        raise ValueError(f'Unsupported text encoder: {text_cfg.hf_model_name}')
    else:
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
