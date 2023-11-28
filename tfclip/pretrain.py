from tfclip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD, INCEPTION_MEAN, INCEPTION_STD, \
    IMAGENET_MEAN, IMAGENET_STD


def _pcfg(version='1.0.0', sha256=None, **kwargs):
    # OpenAI / OpenCLIP defaults
    return {
        'mean': OPENAI_DATASET_MEAN,
        'std': OPENAI_DATASET_STD,
        'interpolation': 'bicubic',
        'resize_mode': 'shortest',
        'version': version,
        'sha256': sha256,
        **kwargs,
    }


def _slpcfg(version='1.0.0', sha256=None, **kwargs):
    # SiGLIP defaults
    return {
        'mean': INCEPTION_MEAN,
        'std': INCEPTION_STD,
        'interpolation': 'bicubic',
        'resize_mode': 'squash',
        'version': version,
        'sha256': sha256,
        **kwargs,
    }


def _apcfg(version='1.0.0', sha256=None, **kwargs):
    # CLIPA defaults
    return {
        'mean': IMAGENET_MEAN,
        'std': IMAGENET_STD,
        'interpolation': 'bilinear',
        'resize_mode': 'squash',
        'version': version,
        'sha256': sha256,
        **kwargs,
    }


# _RN50 = dict(
#     openai=_pcfg(),
#     yfcc15m=_pcfg(),
#     cc12m=_pcfg(),
# )
#
# _RN50_quickgelu = dict(
#     openai=_pcfg(),
#     yfcc15m=_pcfg(),
#     cc12m=_pcfg(),
# )
#
# _RN101 = dict(
#     openai=_pcfg(),
#     yfcc15m=_pcfg(),
# )
#
# _RN101_quickgelu = dict(
#     openai=_pcfg(),
#     yfcc15m=_pcfg(),
# )
#
# _RN50x4 = dict(
#     openai=_pcfg(),
# )
#
# _RN50x16 = dict(
#     openai=_pcfg(),
# )
#
# _RN50x64 = dict(
#     openai=_pcfg(),
# )

_VITB32 = dict(
    # openai=_pcfg(),  # OpenAI models were trained with QuickGELU
    laion400m_e31=_pcfg(),
    laion400m_e32=_pcfg(),
    laion2b_e16=_pcfg(),
    laion2b_s34b_b79k=_pcfg(sha256='c9b3aa9965d2dc9d34fe9751ae3b8b00b56cf30bcf260ffab2cf9c7cb6ecc38a'),
    # DataComp-XL models
    datacomp_xl_s13b_b90k=_pcfg(),
    # DataComp-M models
    datacomp_m_s128m_b4k=_pcfg(),
    commonpool_m_clip_s128m_b4k=_pcfg(),
    commonpool_m_laion_s128m_b4k=_pcfg(),
    commonpool_m_image_s128m_b4k=_pcfg(),
    commonpool_m_text_s128m_b4k=_pcfg(),
    commonpool_m_basic_s128m_b4k=_pcfg(),
    commonpool_m_s128m_b4k=_pcfg(),
    # DataComp-S models
    datacomp_s_s13m_b4k=_pcfg(),
    commonpool_s_clip_s13m_b4k=_pcfg(),
    commonpool_s_laion_s13m_b4k=_pcfg(),
    commonpool_s_image_s13m_b4k=_pcfg(),
    commonpool_s_text_s13m_b4k=_pcfg(),
    commonpool_s_basic_s13m_b4k=_pcfg(),
    commonpool_s_s13m_b4k=_pcfg(),
)

_VITB32_quickgelu = dict(
    openai=_pcfg(sha256='2e0f33c7e468ddaf7e5f615ad557faaaca16ba1f1cf49558953f8bda6cd16006'),
    laion400m_e31=_pcfg(),
    laion400m_e32=_pcfg(),
    metaclip_400m=_pcfg(),
    metaclip_fullcc=_pcfg(),
)

_VITB32_256 = dict(
    datacomp_s34b_b86k=_pcfg(),
)

_VITB16 = dict(
    # openai=_pcfg(),  # OpenAI models were trained with QuickGELU
    laion400m_e31=_pcfg(),
    laion400m_e32=_pcfg(),
    laion2b_s34b_b88k=_pcfg(),
    # DataComp-XL models
    datacomp_xl_s13b_b90k=_pcfg(),
    # DataComp-L models
    datacomp_l_s1b_b8k=_pcfg(),
    commonpool_l_clip_s1b_b8k=_pcfg(),
    commonpool_l_laion_s1b_b8k=_pcfg(),
    commonpool_l_image_s1b_b8k=_pcfg(),
    commonpool_l_text_s1b_b8k=_pcfg(),
    commonpool_l_basic_s1b_b8k=_pcfg(),
    commonpool_l_s1b_b8k=_pcfg(),
)

_VITB16_quickgelu = dict(
    openai=_pcfg(),  # OpenAI models were trained with QuickGELU
    metaclip_400m=_pcfg(),
    metaclip_fullcc=_pcfg(),
)

_VITB16_PLUS_240 = dict(
    laion400m_e31=_pcfg(),
    laion400m_e32=_pcfg(),
)

_VITL14 = dict(
    # openai=_pcfg(),  # OpenAI models were trained with QuickGELU
    laion400m_e31=_pcfg(),
    laion400m_e32=_pcfg(),
    laion2b_s32b_b82k=_pcfg(mean=INCEPTION_MEAN, std=INCEPTION_STD),
    # DataComp-XL models
    datacomp_xl_s13b_b90k=_pcfg(),
    commonpool_xl_clip_s13b_b90k=_pcfg(),
    commonpool_xl_laion_s13b_b90k=_pcfg(),
    commonpool_xl_s13b_b90k=_pcfg(),
)

_VITL14_quickgelu = dict(
    openai=_pcfg(),  # OpenAI models were trained with QuickGELU
    metaclip_400m=_pcfg(),
    metaclip_fullcc=_pcfg(),
)

_VITL14_336_quickgelu = dict(  # OpenAI models were trained with QuickGELU
    openai=_pcfg(),
)

_VITH14 = dict(
    laion2b_s32b_b79k=_pcfg(),
)

_VITH14_quickgelu = dict(
    metaclip_fullcc=_pcfg(),
)

_VITg14 = dict(
    laion2b_s12b_b42k=_pcfg(),
    laion2b_s34b_b88k=_pcfg(),
)

_VITbigG14 = dict(
    laion2b_s39b_b160k=_pcfg(),
)

# _robertaViTB32 = dict(
#     laion2b_s12b_b32k=_pcfg(),
# )
#
# _xlmRobertaBaseViTB32 = dict(
#     laion5b_s13b_b90k=_pcfg(),
# )
#
# _xlmRobertaLargeFrozenViTH14 = dict(
#     frozen_laion5b_s13b_b90k=_pcfg(),
# )
#
# _convnext_base = dict(
#     laion400m_s13b_b51k=_pcfg(),
# )
#
# _convnext_base_w = dict(
#     laion2b_s13b_b82k=_pcfg(),
#     laion2b_s13b_b82k_augreg=_pcfg(),
#     laion_aesthetic_s13b_b82k=_pcfg(),
# )
#
# _convnext_base_w_320 = dict(
#     laion_aesthetic_s13b_b82k=_pcfg(),
#     laion_aesthetic_s13b_b82k_augreg=_pcfg(),
# )
#
# _convnext_large_d = dict(
#     laion2b_s26b_b102k_augreg=_pcfg(),
# )
#
# _convnext_large_d_320 = dict(
#     laion2b_s29b_b131k_ft=_pcfg(),
#     laion2b_s29b_b131k_ft_soup=_pcfg(),
# )
#
# _convnext_xxlarge = dict(
#     laion2b_s34b_b82k_augreg=_pcfg(),
#     laion2b_s34b_b82k_augreg_rewind=_pcfg(),
#     laion2b_s34b_b82k_augreg_soup=_pcfg(),
# )

_coca_VITB32 = dict(
    laion2b_s13b_b90k=_pcfg(sha256='ff9b00049daad95ffc724ae8b57f5760ea99e6f41b7e4ad8b69198198dd6f767'),
    mscoco_finetuned_laion2b_s13b_b90k=_pcfg()
)

_coca_VITL14 = dict(
    laion2b_s13b_b90k=_pcfg(),
    mscoco_finetuned_laion2b_s13b_b90k=_pcfg()
)

_PRETRAINED = {
    # 'RN50': _RN50,
    # 'RN50-quickgelu': _RN50_quickgelu,
    # 'RN101': _RN101,
    # 'RN101-quickgelu': _RN101_quickgelu,
    # 'RN50x4': _RN50x4,
    # 'RN50x16': _RN50x16,
    # 'RN50x64': _RN50x64,

    'ViT-B-32': _VITB32,
    'ViT-B-32-256': _VITB32_256,
    'ViT-B-32-quickgelu': _VITB32_quickgelu,
    'ViT-B-16': _VITB16,
    'ViT-B-16-quickgelu': _VITB16_quickgelu,
    'ViT-B-16-plus-240': _VITB16_PLUS_240,
    'ViT-L-14': _VITL14,
    'ViT-L-14-quickgelu': _VITL14_quickgelu,
    'ViT-L-14-336-quickgelu': _VITL14_336_quickgelu,  # OpenAI models were trained with QuickGELU
    'ViT-H-14': _VITH14,
    'ViT-H-14-quickgelu': _VITH14_quickgelu,
    'ViT-g-14': _VITg14,
    'ViT-bigG-14': _VITbigG14,

    # 'roberta-ViT-B-32': _robertaViTB32,
    # 'xlm-roberta-base-ViT-B-32': _xlmRobertaBaseViTB32,
    # 'xlm-roberta-large-ViT-H-14': _xlmRobertaLargeFrozenViTH14,

    # 'convnext_base': _convnext_base,
    # 'convnext_base_w': _convnext_base_w,
    # 'convnext_base_w_320': _convnext_base_w_320,
    # 'convnext_large_d': _convnext_large_d,
    # 'convnext_large_d_320': _convnext_large_d_320,
    # 'convnext_xxlarge': _convnext_xxlarge,

    'coca_ViT-B-32': _coca_VITB32,
    'coca_ViT-L-14': _coca_VITL14,

    # 'EVA01-g-14': dict(
    #     # from QuanSun/EVA-CLIP/EVA01_CLIP_g_14_psz14_s11B.pt
    #     laion400m_s11b_b41k=_pcfg(),
    # ),
    # 'EVA01-g-14-plus': dict(
    #     # from QuanSun/EVA-CLIP/EVA01_CLIP_g_14_plus_psz14_s11B.pt
    #     merged2b_s11b_b114k=_pcfg(),
    # ),
    # 'EVA02-B-16': dict(
    #     # from QuanSun/EVA-CLIP/EVA02_CLIP_B_psz16_s8B.pt
    #     merged2b_s8b_b131k=_pcfg(),
    # ),
    # 'EVA02-L-14': dict(
    #     # from QuanSun/EVA-CLIP/EVA02_CLIP_L_psz14_s4B.pt
    #     merged2b_s4b_b131k=_pcfg(),
    # ),
    # 'EVA02-L-14-336': dict(
    #     # from QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt
    #     merged2b_s6b_b61k=_pcfg(),
    # ),
    # 'EVA02-E-14': dict(
    #     # from QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_s4B.pt
    #     laion2b_s4b_b115k=_pcfg(),
    # ),
    # 'EVA02-E-14-plus': dict(
    #     # from QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B.pt
    #     laion2b_s9b_b144k=_pcfg(),
    # ),

    'ViT-B-16-SigLIP': dict(
        webli=_slpcfg(sha256='cb71518557f6e047e24d92aeb72e78b519b151dadaed5123d28cb71940cf4c66'),
    ),
    'ViT-B-16-SigLIP-256': dict(
        webli=_slpcfg(),
    ),
    'ViT-B-16-SigLIP-i18n-256': dict(
        webli=_slpcfg(),
    ),
    'ViT-B-16-SigLIP-384': dict(
        webli=_slpcfg(),
    ),
    'ViT-B-16-SigLIP-512': dict(
        webli=_slpcfg(),
    ),
    'ViT-L-16-SigLIP-256': dict(
        webli=_slpcfg(),
    ),
    'ViT-L-16-SigLIP-384': dict(
        webli=_slpcfg(),
    ),
    'ViT-SO400M-14-SigLIP': dict(
        webli=_slpcfg(),
    ),
    'ViT-SO400M-14-SigLIP-384': dict(
        webli=_slpcfg(),
    ),

    'ViT-L-14-CLIPA': dict(
        datacomp1b=_apcfg(sha256='c6b750409035b20d8d43a260a590a643cd2bdce7e00a19074266228e799b36da'),
    ),
    'ViT-L-14-CLIPA-336': dict(
        datacomp1b=_apcfg(),
    ),
    'ViT-H-14-CLIPA': dict(
        datacomp1b=_apcfg(),
    ),
    'ViT-H-14-CLIPA-336-quickgelu': dict(
        laion2b=_apcfg(),
        datacomp1b=_apcfg(),
    ),
    'ViT-bigG-14-CLIPA': dict(
        datacomp1b=_apcfg(),
    ),
    'ViT-bigG-14-CLIPA-336': dict(
        datacomp1b=_apcfg(),
    ),

    # 'nllb-clip-base': dict(
    #     v1=_pcfg(),
    # ),
    # 'nllb-clip-large': dict(
    #     v1=_pcfg(),
    # )
}


def _clean_tag(tag):
    return tag.lower().replace('-', '_')


def list_pretrained(as_str=False):
    return [':'.join([k, t]) if as_str else (k, t) for k in _PRETRAINED.keys() for t in _PRETRAINED[k].keys()]


def list_pretrained_models_by_tag(tag):
    models = []
    tag = _clean_tag(tag)
    for k in _PRETRAINED.keys():
        if tag in _PRETRAINED[k]:
            models.append(k)

    return models


def list_pretrained_tags_by_model(model):
    tags = []
    if model in _PRETRAINED:
        tags.extend(_PRETRAINED[model].keys())
    return tags


def get_pretrained_cfg(model, tag):
    if model not in _PRETRAINED:
        return {}
    if tag is None:
        return {}

    model_pretrained = _PRETRAINED[model]

    return model_pretrained.get(_clean_tag(tag), {})


def get_pretrained_url(model, tag):
    tag = _clean_tag(tag)
    cfg = get_pretrained_cfg(model, tag)

    version = cfg.get('version', None)
    if version is None:
        return None

    sha256 = cfg.get('sha256', None)
    if sha256 is None:
        return None

    return f'https://github.com/shkarupa-alex/tfclip/releases/download/{version}/{model}__{tag}__{sha256}.h5'
