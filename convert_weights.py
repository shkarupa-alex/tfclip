#!/usr/bin/env python3
import sys

if '/Users/alex/Downloads/open_clip-main/src' not in sys.path:
    sys.path.append('/Users/alex/Downloads/open_clip-main/src')

import argparse
import os
import open_clip
import tfclip


def transform_weights(weights, embeds, heads):
    for key in list(weights.keys()):
        embed = embeds[0] if key.startswith('text/') else embeds[1]
        head = heads[0] if key.startswith('text/') else heads[1]
        head_dim = embed // head

        value = weights[key]

        if 'text/token/pos/embedding:0' in key:
            value = value[None]

        if 'vision/patch/embed/kernel:0' in key:
            value = value.transpose(2, 3, 1, 0)

        if 'vision/patch/cls/token:0' in key:
            value = value[None, None]

        if 'vision/patch/pos/embedding:0' in key:
            value = value[None]

        if '/scale/' in key:
            value = value[None, None]

        if any([part in key for part in {'/value/', '/attention_output/', '/expand/', '/squeeze/', '/proj/'}]):
            value = value.T

        if '/value/bias:0' in key:
            value = value.reshape(3, head, head_dim)
            weights[key.replace('/value/', '/query/')] = value[0]
            weights[key.replace('/value/', '/key/')] = value[1]
            value = value[2]

        if '/value/kernel:0' in key:
            value = value.reshape(embed, 3, head, head_dim)
            weights[key.replace('/value/', '/query/')] = value[:, 0]
            weights[key.replace('/value/', '/key/')] = value[:, 1]
            value = value[:, 2]

        if '/attention_output/kernel:0' in key:
            value = value.reshape(head, head_dim, embed)

        if 'vision/head/proj/kernel' in key:
            value = value.T

        if 'text/head/proj/kernel' in key:
            value = value.T

        weights[key] = value

    return weights


def convert_name(n):
    n = f'{n}:0' if n.startswith('visual') else f'text/{n}:0'
    n = n.replace('visual.', 'vision/').replace('transformer.resblocks.', 'layer_')

    n = n.replace('.ln_1.', '/attn/norm/').replace('.ln_2.', '/mlp/norm/')
    n = n.replace('/ln_pre.', '/patch/norm/').replace('/ln_final.', '/head/norm/').replace('/ln_post.', '/head/norm/')

    n = n.replace('.attn.in_proj_', '/attn/mhsa/value/').replace('.attn.out_proj.', '/attn/mhsa/attention_output/')
    n = n.replace('.mlp.c_fc.', '/mlp/expand/').replace('.mlp.c_proj.', '/mlp/squeeze/')

    n = n.replace('text/token_embedding.weight', 'text/token/embed/embeddings')
    n = n.replace('text/positional_embedding', 'text/token/pos/embedding')
    n = n.replace('text/text_projection', 'text/head/proj/kernel')

    n = n.replace('vision/conv1.weight', 'vision/patch/embed/kernel')
    n = n.replace('vision/class_embedding', 'vision/patch/cls/token')
    n = n.replace('vision/positional_embedding', 'vision/patch/pos/embedding')
    n = n.replace('vision/proj', 'vision/head/proj/kernel')

    n = n.replace('text/logit_scale', 'head/sim/scale')

    n = n.replace('/norm/weight', '/norm/gamma').replace('/norm/bias', '/norm/beta').replace('/weight', '/kernel')

    return n


if '__main__' == __name__:
    clip_pretrained = open_clip.list_pretrained()
    pretrained_models = list(set([p[0] for p in clip_pretrained]))
    pretrained_weights = list(set([p[1] for p in clip_pretrained]))

    parser = argparse.ArgumentParser(description='Swin Transformer weight conversion from PyTorch to TensorFlow')
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

    if (argv.model_name, argv.model_pretrain) not in clip_pretrained:
        allowed_weights = list(set([p[1] for p in clip_pretrained if p[0] == argv.model_name]))
        raise ValueError(
            f'Required combination of model and weights not available. '
            f'Available weights for {argv.model_name} are: {allowed_weights}')

    model_tf = tfclip.create_model(argv.model_name, pretrained=None)
    text_embed, text_heads = model_tf.get_layer(name='text/layer_0/attn/mhsa')._query_dense.kernel.shape[:2]
    vision_embed, vision_heads = model_tf.get_layer(name='vision/layer_0/attn/mhsa')._query_dense.kernel.shape[:2]

    model_torch, _, _ = open_clip.create_model_and_transforms(argv.model_name, pretrained=argv.model_pretrain)
    weights_torch = model_torch.state_dict()
    weights_torch = {convert_name(k): v.numpy() for k, v in weights_torch.items()}
    weights_torch = transform_weights(weights_torch, (text_embed, vision_embed), (text_heads, vision_heads))

    # wtd = sorted([(k, v.shape) for k, v in weights_torch.items()], key=lambda x: x[0])
    # for x in wtd:
    #     print(x[0], x[1])
    # wtd = sorted([(w.name, w.numpy().shape) for w in model_tf.weights], key=lambda x: x[0])
    # for x in wtd:
    #     print(x[0], x[1])

    weights_tf = []
    for w in model_tf.weights:
        assert w.name in weights_torch, f'Can\'t find weight {w.name} in checkpoint'

        weight = weights_torch[w.name]
        assert w.shape == weight.shape, f'Weight {w.name} shapes not compatible: {w.shape} vs {weight.shape}'

        weights_tf.append(weight)

    model_tf.set_weights(weights_tf)
    model_tf.save_weights(f'{argv.out_path}/{argv.model_name}__{argv.model_pretrain}.h5', save_format='h5')
