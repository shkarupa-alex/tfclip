#!/usr/bin/env python3
import argparse
import numpy as np
import open_clip
import os
import tfclip
from keras.src.utils import data_utils


def transform_weights(weights, embeds, heads):
    if 'vision/head/mapool/mhsa/kv/kernel:0' in weights and 'vision/head/mapool/mhsa/query/bias:0' in weights:
        weights['vision/head/mapool/mhsa/qkv/kernel:0'] = np.concatenate([
            weights.pop('vision/head/mapool/mhsa/query/kernel:0'),
            weights.pop('vision/head/mapool/mhsa/kv/kernel:0')], axis=0)
        weights['vision/head/mapool/mhsa/qkv/bias:0'] = np.concatenate([
            weights.pop('vision/head/mapool/mhsa/query/bias:0'),
            weights.pop('vision/head/mapool/mhsa/kv/bias:0')], axis=0)

    for key in list(weights.keys()):
        if 'text_decoder' in key:
            weights.pop(key)
            continue

        if '/attnpool/' in key or '/mapool/' in key:
            embed, head = embeds[2], heads[2]
        elif key.startswith('vision/'):
            embed, head = embeds[1], heads[1]
        else:
            embed, head = embeds[0], heads[0]
        head_dim = embed // head

        value = weights[key]

        if 'vision/patch/embed/kernel:0' in key:
            value = value.transpose(2, 3, 1, 0)

        if '/pos/embedding:0' in key or '/attnpool/query:0' in key:
            value = value.reshape([d for d in value.shape if d > 1])[None]

        if '/cls/token:0' in key or '/scale/' in key:
            value = value.reshape([d for d in value.shape if d > 1])[None, None]

        if any([part in key for part in {'/qkv/', '/attention_output/', '/expand/', '/squeeze/', '/gate/', '/proj/'}]):
            value = value.T

        if 'vision/head/proj/kernel' in key and 'vision/head/proj/bias:0' not in weights:
            value = value.T

        if 'text/head/proj/kernel' in key and 'text/head/proj/bias:0' not in weights:
            value = value.T

        if '/attention_output/kernel:0' in key:
            value = value.reshape(head, head_dim, embed)

        if any([f'/attnpool/mhsa/{part}/' in key for part in {'query', 'key', 'value'}]):
            value = value.T.reshape(-1, head, head_dim)

        if any([f'/mapool/mhsa/{part}/' in key for part in {'query', 'key', 'value'}]):
            value = value.T.reshape(-1, head, head_dim)

        if any([f'/attn/mhsa/{part}/' in key for part in {'query', 'key', 'value'}]):
            if '/bias' in key:
                value = value.T.reshape(head, head_dim)
            else:
                value = value.T.reshape(-1, head, head_dim)

        if '/qkv/bias:0' in key:
            value = value.reshape(3, head, head_dim)
            weights[key.replace('/qkv/', '/query/')] = value[0]
            weights[key.replace('/qkv/', '/key/')] = value[1]
            weights[key.replace('/qkv/', '/value/')] = value[2]
            weights.pop(key)
            continue

        if '/qkv/kernel:0' in key:
            value = value.reshape(embed, 3, head, head_dim)
            weights[key.replace('/qkv/', '/query/')] = value[:, 0]
            weights[key.replace('/qkv/', '/key/')] = value[:, 1]
            weights[key.replace('/qkv/', '/value/')] = value[:, 2]
            weights.pop(key)
            continue

        weights[key] = value

    for key in list(weights.keys()):
        if 'attn/mhsa/query/bias' not in key:
            continue

        key_ = key.replace('/query/bias', '/key/bias')
        if key_ not in weights:
            weights[key_] = np.zeros_like(weights[key])

    return weights


def convert_name(n):
    if not (n.startswith('visual.') or n.startswith('text.')):
        n = f'text/{n}:0'
    else:
        n = f'{n}:0'

    n = n.replace('visual.trunk.blocks.', 'visual.transformer.resblocks.')  # timm -> open_clip
    n = n.replace('visual.', 'vision/').replace('text.', 'text/').replace('transformer.resblocks.', 'layer_')

    n = n.replace('attn_pool.', 'head/attnpool/')
    n = n.replace('attnpool/ln_k.weight', 'attnpool/ln_k/gamma').replace('attnpool/ln_k.bias', 'attnpool/ln_k/beta')
    n = n.replace('attnpool/ln_q.weight', 'attnpool/ln_q/gamma').replace('attnpool/ln_q.bias', 'attnpool/ln_q/beta')
    n = n.replace('attnpool/attn.q_proj_', 'attnpool/mhsa/query/')
    n = n.replace('attnpool/attn.k_proj_', 'attnpool/mhsa/key/')
    n = n.replace('attnpool/attn.v_proj_', 'attnpool/mhsa/value/')
    n = n.replace('attnpool/attn.in_proj_bias', 'attnpool/mhsa/qkv/bias')
    n = n.replace('attnpool/attn.out_proj.', 'attnpool/mhsa/attention_output/')

    n = n.replace('.ln_1.', '/attn/norm/').replace('.ln_2.', '/mlp/norm/')
    n = n.replace('.norm1.', '/attn/norm/').replace('.norm2.', '/mlp/norm/')
    n = n.replace('/ln_pre.', '/patch/norm/').replace('/ln_final.', '/head/norm/').replace('/ln_post.', '/head/norm/')
    n = n.replace('.attn.in_proj_', '/attn/mhsa/qkv/').replace('.attn.out_proj.', '/attn/mhsa/attention_output/')
    n = n.replace('.attn.qkv.', '/attn/mhsa/qkv/').replace('.attn.proj.', '/attn/mhsa/attention_output/')
    n = n.replace('.attn.q_proj.', '/attn/mhsa/query/').replace('.attn.k_proj.', '/attn/mhsa/key/')
    n = n.replace('.attn.v_proj.', '/attn/mhsa/value/').replace('.attn.norm.weight', '/attn/mhsa/attention_norm/gamma')
    n = n.replace('.attn.norm.bias', '/attn/mhsa/attention_norm/beta')
    n = n.replace('.attn.q_bias', '/attn/mhsa/query/bias').replace('.attn.v_bias', '/attn/mhsa/value/bias')
    n = n.replace('.mlp.c_fc.', '/mlp/expand/').replace('.mlp.c_proj.', '/mlp/squeeze/')
    n = n.replace('.mlp.fc1.', '/mlp/expand/').replace('.mlp.fc2.', '/mlp/squeeze/')
    n = n.replace('.mlp.fc1_x.', '/mlp/expand/').replace('.mlp.fc1_g.', '/mlp/gate/')
    n = n.replace('.mlp.norm.weight', '/mlp/normalize/gamma').replace('.mlp.norm.bias', '/mlp/normalize/beta')

    n = n.replace('text/cls_emb', 'text/token/cls/token')
    n = n.replace('text/token_embedding.weight', 'text/token/embed/embeddings')
    n = n.replace('text/positional_embedding', 'text/token/pos/embedding')
    n = n.replace('text/text_projection.weight', 'text/head/proj/kernel')
    n = n.replace('text/text_projection.bias', 'text/head/proj/bias')
    n = n.replace('text/text_projection', 'text/head/proj/kernel')

    n = n.replace('vision/conv1.weight', 'vision/patch/embed/kernel')
    n = n.replace('vision/class_embedding', 'vision/patch/cls/token')
    n = n.replace('vision/trunk.cls_token', 'vision/patch/cls/token')
    n = n.replace('vision/positional_embedding', 'vision/patch/pos/embedding')
    n = n.replace('vision/proj', 'vision/head/proj/kernel')
    n = n.replace('vision/trunk.head.', 'vision/head/proj/')

    n = n.replace('vision/trunk.patch_embed.proj.', 'vision/patch/embed/')
    n = n.replace('vision/trunk.pos_embed', 'vision/patch/pos/embedding')
    n = n.replace('vision/trunk.head/attnpool/', 'vision/head/mapool/')
    n = n.replace('vision/trunk.norm.', 'vision/head/norm/')
    n = n.replace('mapool/latent', 'mapool/probe')
    n = n.replace('mapool/kv.', 'mapool/mhsa/kv/').replace('mapool/q.', 'mapool/mhsa/query/')
    n = n.replace('mapool/proj.', 'mapool/mhsa/attention_output/')
    n = n.replace('mapool/mlp.fc1.', 'mapool/mlp/expand/').replace('mapool/mlp.fc2.', 'mapool/mlp/squeeze/')
    n = n.replace('mapool/norm.', 'mapool/mlp/norm/')

    n = n.replace('text/logit_scale', 'head/sim/scale')
    n = n.replace('text/logit_bias', 'head/sim/bias')

    n = n.replace('/norm/weight', '/norm/gamma').replace('/norm/bias', '/norm/beta').replace('/weight', '/kernel')

    return n


if '__main__' == __name__:
    clip_pretrained = tfclip.list_pretrained()
    pretrained_models = list(set([p[0] for p in clip_pretrained]))
    pretrained_weights = list(set([p[1] for p in clip_pretrained]))

    parser = argparse.ArgumentParser(description='CLIP weight conversion from PyTorch to TensorFlow')
    parser.add_argument(
        'model_name',
        type=str,
        choices=pretrained_models,
        help='Model architecture to load')
    parser.add_argument(
        'model_pretrain',
        type=str,
        choices=pretrained_weights,
        help='Model checkpoint to load')
    parser.add_argument(
        'out_path',
        type=str,
        help='Path to save TensorFlow model weights')

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.out_path) and os.path.isdir(argv.out_path), 'Wrong output path'

    allowed_weights = tfclip.list_pretrained_tags_by_model(argv.model_name)
    if argv.model_pretrain not in allowed_weights:
        raise ValueError(
            f'Required combination of model and weights is not available. '
            f'Available weights for {argv.model_name} are: {allowed_weights}')

    model_tf, _, _ = tfclip.create_model_and_transforms(argv.model_name, pretrained=None)
    text_embed, text_heads = model_tf.get_layer(name='text/layer_0/attn/mhsa')._query_dense.kernel.shape[:2]
    vision_embed, vision_heads = model_tf.get_layer(name='vision/layer_0/attn/mhsa')._query_dense.kernel.shape[:2]

    try:
        pool_layer = model_tf.get_layer(name='vision/head/attnpool')
        pool_embed, pool_heads = pool_layer.mhsa._query_dense.kernel.shape[:2]
    except ValueError:
        try:
            pool_layer = model_tf.get_layer(name='vision/head/mapool')
            pool_embed, pool_heads = pool_layer.mhsa._query_dense.kernel.shape[:2]
        except ValueError:
            pool_embed, pool_heads = (0, 0)

    # OpenAI models were trained with QuickGELU, but pretrains placed in wrong models (without `-quickgelu`)
    oc_name = argv.model_name.replace('-quickgelu', '') if 'openai' == argv.model_pretrain else argv.model_name
    model_torch = open_clip.create_model(oc_name, pretrained=argv.model_pretrain)
    weights_torch = model_torch.state_dict()
    weights_torch = {convert_name(k): v.numpy() for k, v in weights_torch.items()}
    weights_torch = transform_weights(
        weights_torch, (text_embed, vision_embed, pool_embed), (text_heads, vision_heads, pool_heads))

    weights_tf = []
    for w in model_tf.weights:
        assert w.name in weights_torch, f'Can\'t find weight {w.name} in checkpoint'

        weight = weights_torch.pop(w.name)
        assert w.shape == weight.shape, f'Weight {w.name} shapes not compatible: {w.shape} vs {weight.shape}'

        weights_tf.append(weight)

    if len(weights_torch.keys()):
        raise ValueError(f'Some of original weights did not consumed: {weights_torch.keys()}')
    else:
        model_tf.set_weights(weights_tf)

        weights_name = f'{argv.out_path}/{argv.model_name}__{argv.model_pretrain}.h5'
        model_tf.save_weights(weights_name, save_format='h5')

        weights_hash = data_utils._hash_file(weights_name)
        os.rename(weights_name, weights_name.replace('.h5', f'__{weights_hash}.h5'))
