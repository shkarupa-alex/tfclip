import json
import re
from copy import deepcopy
from dataclasses import asdict
from keras.src.utils import data_utils
from pathlib import Path
from tfclip.tokenizer import SimpleTokenizer, DEFAULT_CONTEXT_LENGTH
from tfclip.transform import PreprocessCfg, merge_preprocess_dict
from tfclip.model import CLIP
from tfclip.pretrained import get_pretrained_cfg

HF_HUB_PREFIX = 'hf-hub:'
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
    return list(_MODEL_CONFIGS.keys())


def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def get_tokenizer(model_name='', context_length=None, **kwargs):
    if model_name.startswith(HF_HUB_PREFIX):
        raise ValueError(f'Unsupported text tokenizer: {model_name}')
    else:
        config = get_model_config(model_name)
        if config is None:
            raise ValueError(f'No valid model config found for {model_name}')

    text_config = config.get('text_cfg', {})
    if 'tokenizer_kwargs' in text_config:
        tokenizer_kwargs = dict(text_config['tokenizer_kwargs'], **kwargs)
    else:
        tokenizer_kwargs = kwargs

    if context_length is None:
        context_length = text_config.get('context_length', DEFAULT_CONTEXT_LENGTH)

    if 'hf_tokenizer_name' in text_config:
        raise ValueError(f'Unsupported text tokenizer: {text_config["hf_tokenizer_name"]}')
    else:
        tokenizer = SimpleTokenizer(context_length=context_length, **tokenizer_kwargs)

    return tokenizer


def create_model(model_name, pretrained=None, **model_kwargs):
    if model_name.startswith(HF_HUB_PREFIX):
        raise ValueError(f'Unsupported model: {model_name}')
    else:
        model_name = model_name.replace('/', '-')
        model_cfg = None

    preprocess_cfg = asdict(PreprocessCfg())
    if pretrained is not None:
        pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
        preprocess_cfg = merge_preprocess_dict(preprocess_cfg, pretrained_cfg)

    if pretrained and pretrained.lower() == 'openai':
        raise ValueError(f'Unsupported model weights: openai')
    else:
        model_cfg = model_cfg or get_model_config(model_name)
        if model_cfg is None:
            raise ValueError(f'Model config for {model_name} not found')

        if 'timm_model_name' in model_cfg.get('vision_cfg', {}):
            raise ValueError(f'Unsupported image encoder: {model_cfg["vision_cfg"]["timm_model_name"]}')

        if 'hf_model_name' in model_cfg.get('text_cfg', {}):
            raise ValueError(f'Unsupported text encoder: {model_cfg["text_cfg"]["hf_model_name"]}')

        custom_text = model_cfg.pop('custom_text', False)

        model_cfg = dict(model_cfg, **model_kwargs)
        if custom_text:
            if 'multimodal_cfg' in model_cfg:
                raise ValueError('Custom multimodal text encoders not supported')
            else:
                model = CLIP(
                    **model_cfg, img_mean=preprocess_cfg['mean'], img_std=preprocess_cfg['std'], custom_text=True)
        else:
            model = CLIP(**model_cfg, img_mean=preprocess_cfg['mean'], img_std=preprocess_cfg['std'])

        # Load weights.
        if pretrained in {}:  # TODO
            weights_url = ''
            weights_hash = ''
            weights_path = data_utils.get_file(origin=weights_url, file_hash=weights_hash, cache_subdir='tfvit')
            model.load_weights(weights_path)
        elif pretrained is not None:
            model.load_weights(pretrained)

    return model
