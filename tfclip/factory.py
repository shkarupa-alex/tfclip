import json
import re
from copy import deepcopy
from dataclasses import asdict
from keras.src.utils import data_utils
from pathlib import Path
from tfclip.tokenizer import DEFAULT_CONTEXT_LENGTH, SimpleTokenizer, SentencePieceTokenizer, TensorflowHubTokenizer
from tfclip.transform import PreprocessCfg, merge_preprocess_dict, image_transform
from tfclip.model import CLIP
from tfclip.pretrain import get_pretrained_cfg, list_pretrained_tags_by_model, get_pretrained_url

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f'configs/']
_MODEL_CONFIGS = {}


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    global _MODEL_CONFIGS

    return list(_MODEL_CONFIGS.keys())


def get_model_config(model_name):
    global _MODEL_CONFIGS

    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def get_tokenizer(model_name='', context_length=None, **kwargs):
    config = get_model_config(model_name)
    if config is None:
        raise ValueError(f'No valid model config found for {model_name}')

    text_config = config.get('text_cfg', {})
    tokenizer_kwargs = dict(text_config.get('tokenizer_kwargs', {}), **kwargs)

    if context_length is None:
        context_length = text_config.get('context_length', DEFAULT_CONTEXT_LENGTH)
    tokenizer_kwargs['context_length'] = context_length

    if 'sp_tokenizer_name' in text_config:
        tokenizer = SentencePieceTokenizer(text_config['sp_tokenizer_name'], **tokenizer_kwargs)
    elif 'hub_tokenizer_name' in text_config:
        tokenizer = TensorflowHubTokenizer(text_config['hub_tokenizer_name'], **tokenizer_kwargs)
    else:
        tokenizer = SimpleTokenizer(**tokenizer_kwargs)

    return tokenizer


def create_model_and_transforms(model_name, pretrained=None, weights_path=None, **model_kwargs):
    model_name = model_name.replace('/', '-')

    model_cfg = get_model_config(model_name)
    if model_cfg is None:
        raise ValueError(f'Model config for {model_name} not found')

    allowed_weights = list_pretrained_tags_by_model(model_name)
    if pretrained is not None and pretrained not in allowed_weights:
        raise ValueError(
            f'Required combination of model and weights is not available. '
            f'Available weights for {model_name} are: {allowed_weights}')

    preprocess_cfg = asdict(PreprocessCfg())
    if pretrained is not None:
        pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
        preprocess_cfg = merge_preprocess_dict(preprocess_cfg, pretrained_cfg)

    bias_init = model_cfg.pop('init_logit_bias', None)
    custom_text = model_cfg.pop('custom_text', False)

    model_cfg = dict(model_cfg, **model_kwargs)
    model_cfg.pop('multimodal_cfg', None)

    model = CLIP(
        **model_cfg, img_mean=preprocess_cfg['mean'], img_std=preprocess_cfg['std'], bias_init=bias_init,
        custom_text=custom_text)

    if pretrained is not None:
        if weights_path is None:
            weights_url = get_pretrained_url(model_name, pretrained)
            if weights_url is None:
                raise ValueError(
                    f'Pretrained weighs "{pretrained}" for model "{model_name}" are not ready and can\'t be '
                    f'downloaded. You can convert them locally with `convert_weights.py` script (open_clip should be '
                    f'installed) and supply with `weights_path` argument.')

            weights_hash = weights_url.split('__')[-1].replace('.h5', '')
            weights_path = data_utils.get_file(origin=weights_url, file_hash=weights_hash, cache_subdir='tfclip')
        model.load_weights(weights_path)

    image_prep = image_transform(
        model_cfg['vision_cfg']['image_size'], preprocess_cfg['interpolation'], preprocess_cfg['resize_mode'])
    text_prep = get_tokenizer(model_name)

    return model, image_prep, text_prep
