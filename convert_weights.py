#!/usr/bin/env python3
import argparse
import os

import numpy as np
import open_clip
from keras.src.utils.file_utils import hash_file

import tfclip


def transform_weights(weights, embeds, heads):
    if (
        "vision_head_mapool/mhsa/kv/kernel" in weights
        and "vision_head_mapool/mhsa/query/bias" in weights
    ):
        weights["vision_head_mapool/mhsa/qkv/kernel"] = np.concatenate(
            [
                weights.pop("vision_head_mapool/mhsa/query/kernel"),
                weights.pop("vision_head_mapool/mhsa/kv/kernel"),
            ],
            axis=0,
        )
        weights["vision_head_mapool/mhsa/qkv/bias"] = np.concatenate(
            [
                weights.pop("vision_head_mapool/mhsa/query/bias"),
                weights.pop("vision_head_mapool/mhsa/kv/bias"),
            ],
            axis=0,
        )

    for key in list(weights.keys()):
        if "text_decoder" in key:
            weights.pop(key)
            continue

        if "_attnpool/" in key or "_mapool/" in key:
            embed, head = embeds[2], heads[2]
        elif key.startswith("vision_"):
            embed, head = embeds[1], heads[1]
        else:
            embed, head = embeds[0], heads[0]
        head_dim = embed // head

        value = weights[key]

        if "vision_patch_embed/kernel" in key:
            value = value.transpose(2, 3, 1, 0)

        if "_pos/embedding" in key or "_attnpool/query" in key:
            value = value.reshape([d for d in value.shape if d > 1])[None]

        if "_cls/token" in key or "_scale/" in key:
            value = value.reshape([d for d in value.shape if d > 1])[None, None]

        if any(
            [
                part in key
                for part in {
                    "/qkv/",
                    "/attention_output/",
                    "_expand/",
                    "_squeeze/",
                    "_gate/",
                    "_proj/",
                }
            ]
        ):
            value = value.T

        if (
            "vision_head_proj/kernel" in key
            and "vision_head_proj/bias" not in weights
        ):
            value = value.T

        if (
            "text_head_proj/kernel" in key
            and "text_head_proj/bias" not in weights
        ):
            value = value.T

        if "/attention_output/kernel" in key:
            value = value.reshape(head, head_dim, embed)

        if any(
            [
                f"_attnpool/mhsa/{part}/" in key
                for part in {"query", "key", "value"}
            ]
        ):
            value = value.T.reshape(-1, head, head_dim)

        if any(
            [
                f"_mapool/mhsa/{part}/" in key
                for part in {"query", "key", "value"}
            ]
        ):
            value = value.T.reshape(-1, head, head_dim)

        if any(
            [f"_attn_mhsa/{part}/" in key for part in {"query", "key", "value"}]
        ):
            if "/bias" in key:
                value = value.T.reshape(head, head_dim)
            else:
                value = value.T.reshape(-1, head, head_dim)

        if "/qkv/bias" in key:
            value = value.reshape(3, head, head_dim)
            weights[key.replace("/qkv/", "/query/")] = value[0]
            weights[key.replace("/qkv/", "/key/")] = value[1]
            weights[key.replace("/qkv/", "/value/")] = value[2]
            weights.pop(key)
            continue

        if "/qkv/kernel" in key:
            value = value.reshape(embed, 3, head, head_dim)
            weights[key.replace("/qkv/", "/query/")] = value[:, 0]
            weights[key.replace("/qkv/", "/key/")] = value[:, 1]
            weights[key.replace("/qkv/", "/value/")] = value[:, 2]
            weights.pop(key)
            continue

        weights[key] = value

    for key in list(weights.keys()):
        if "attn_mhsa/query/bias" not in key:
            continue

        key_ = key.replace("/query/bias", "/key/bias")
        if key_ not in weights:
            weights[key_] = np.zeros_like(weights[key])

    return weights


def convert_name(n):
    if not (n.startswith("visual.") or n.startswith("text.")):
        n = f"text_{n}"

    n = n.replace(
        "visual.trunk.blocks.", "visual.transformer.resblocks."
    )  # timm -> open_clip
    n = (
        n.replace("visual.", "vision_")
        .replace("text.", "text_")
        .replace("transformer.resblocks.", "layer_")
    )

    n = n.replace("attn_pool.", "head_attnpool/")
    n = n.replace("attnpool/ln_k.weight", "attnpool/ln_k/gamma").replace(
        "attnpool/ln_k.bias", "attnpool/ln_k/beta"
    )
    n = n.replace("attnpool/ln_q.weight", "attnpool/ln_q/gamma").replace(
        "attnpool/ln_q.bias", "attnpool/ln_q/beta"
    )
    n = n.replace("attnpool/attn.q_proj_", "attnpool/mhsa/query/")
    n = n.replace("attnpool/attn.k_proj_", "attnpool/mhsa/key/")
    n = n.replace("attnpool/attn.v_proj_", "attnpool/mhsa/value/")
    n = n.replace("attnpool/attn.in_proj_bias", "attnpool/mhsa/qkv/bias")
    n = n.replace("attnpool/attn.out_proj.", "attnpool/mhsa/attention_output/")

    n = n.replace(".ln_1.", "_attn_norm/").replace(".ln_2.", "_mlp_norm/")
    n = n.replace(".norm1.", "_attn_norm/").replace(".norm2.", "_mlp_norm/")
    n = (
        n.replace("_ln_pre.", "_patch_norm/")
        .replace("_ln_final.", "_head_norm/")
        .replace("_ln_post.", "_head_norm/")
    )
    n = n.replace(".attn.in_proj_", "_attn_mhsa/qkv/").replace(
        ".attn.out_proj.", "_attn_mhsa/attention_output/"
    )
    n = n.replace(".attn.qkv.", "_attn_mhsa/qkv/").replace(
        ".attn.proj.", "_attn_mhsa/attention_output/"
    )
    n = n.replace(".attn.q_proj.", "_attn_mhsa/query/").replace(
        ".attn.k_proj.", "_attn_mhsa/key/"
    )
    n = n.replace(".attn.v_proj.", "_attn_mhsa/value/").replace(
        ".attn.norm.weight", "_attn_mhsa/attention_norm/gamma"
    )
    n = n.replace(".attn.norm.bias", "_attn_mhsa/attention_norm/beta")
    n = n.replace(".attn.q_bias", "_attn_mhsa/query/bias").replace(
        ".attn.v_bias", "_attn_mhsa/value/bias"
    )
    n = n.replace(".mlp.c_fc.", "_mlp_expand/").replace(
        ".mlp.c_proj.", "_mlp_squeeze/"
    )
    n = n.replace(".mlp.fc1.", "_mlp_expand/").replace(
        ".mlp.fc2.", "_mlp_squeeze/"
    )
    n = n.replace(".mlp.fc1_x.", "_mlp_expand/").replace(
        ".mlp.fc1_g.", "_mlp_gate/"
    )
    n = n.replace(".mlp.norm.weight", "_mlp_normalize/gamma").replace(
        ".mlp.norm.bias", "_mlp_normalize/beta"
    )

    n = n.replace("text_cls_emb", "text_token_cls/token")
    n = n.replace("text_token_embedding.weight", "text_token_embed/embeddings")
    n = n.replace("text_positional_embedding", "text_token_pos/embedding")
    n = n.replace("text_text_projection.weight", "text_head_proj/kernel")
    n = n.replace("text_text_projection.bias", "text_head_proj/bias")
    n = n.replace("text_text_projection", "text_head_proj/kernel")

    n = n.replace("vision_conv1.weight", "vision_patch_embed/kernel")
    n = n.replace("vision_class_embedding", "vision_patch_cls/token")
    n = n.replace("vision_trunk.cls_token", "vision_patch_cls/token")
    n = n.replace("vision_positional_embedding", "vision_patch_pos/embedding")
    n = n.replace("vision_proj", "vision_head_proj/kernel")
    n = n.replace("vision_trunk.head.", "vision_head_proj/")

    n = n.replace("vision_trunk.patch_embed.proj.", "vision_patch_embed/")
    n = n.replace("vision_trunk.pos_embed", "vision_patch_pos/embedding")
    n = n.replace("vision_trunk.head_attnpool/", "vision_head_mapool/")
    n = n.replace("vision_trunk.norm.", "vision_head_norm/")
    n = n.replace("mapool/latent", "mapool/probe")
    n = n.replace("mapool/kv.", "mapool/mhsa/kv/").replace(
        "mapool/q.", "mapool/mhsa/query/"
    )
    n = n.replace("mapool/proj.", "mapool/mhsa/attention_output/")
    n = n.replace("mapool/mlp.fc1.", "mapool_mlp_expand/").replace(
        "mapool/mlp.fc2.", "mapool_mlp_squeeze/"
    )
    n = n.replace("mapool/norm.", "mapool_mlp_norm/")

    n = n.replace("text_logit_scale", "head_sim/scale")
    n = n.replace("text_logit_bias", "head_sim/bias")

    n = (
        n.replace("_norm/weight", "_norm/gamma")
        .replace("_norm/bias", "_norm/beta")
        .replace("/weight", "/kernel")
    )

    return n


if "__main__" == __name__:
    clip_pretrained = tfclip.list_pretrained()
    pretrained_models = list(set([p[0] for p in clip_pretrained]))
    pretrained_weights = list(set([p[1] for p in clip_pretrained]))

    parser = argparse.ArgumentParser(
        description="CLIP weight conversion from PyTorch to TensorFlow"
    )
    parser.add_argument(
        "model_name",
        type=str,
        choices=pretrained_models,
        help="Model architecture to load",
    )
    parser.add_argument(
        "model_pretrain",
        type=str,
        choices=pretrained_weights,
        help="Model checkpoint to load",
    )
    parser.add_argument(
        "out_path", type=str, help="Path to save TensorFlow model weights"
    )

    argv, _ = parser.parse_known_args()
    if not os.path.exists(argv.out_path) or not os.path.isdir(argv.out_path):
        raise ValueError(
            f"Output path does not exist or "
            f"is not a directory: {argv.out_path}"
        )

    allowed_weights = tfclip.list_pretrained_tags_by_model(argv.model_name)
    if argv.model_pretrain not in allowed_weights:
        raise ValueError(
            f"Required combination of model and weights is not available. "
            f"Available weights for {argv.model_name} are: {allowed_weights}"
        )

    model_tf, _, _ = tfclip.create_model_and_transforms(
        argv.model_name, pretrained=None
    )
    text_embed, text_heads = model_tf.get_layer(
        name="text_layer_0_attn_mhsa"
    )._query_dense.kernel.shape[:2]
    vision_embed, vision_heads = model_tf.get_layer(
        name="vision_layer_0_attn_mhsa"
    )._query_dense.kernel.shape[:2]

    try:
        pool_layer = model_tf.get_layer(name="vision_head_attnpool")
        pool_embed, pool_heads = pool_layer.mhsa._query_dense.kernel.shape[:2]
    except ValueError:
        try:
            pool_layer = model_tf.get_layer(name="vision_head_mapool")
            pool_embed, pool_heads = pool_layer.mhsa._query_dense.kernel.shape[
                :2
            ]
        except ValueError:
            pool_embed, pool_heads = (0, 0)

    # OpenAI models were trained with QuickGELU,
    # but pretrains placed in wrong models (without `-quickgelu`)
    oc_name = (
        argv.model_name.replace("-quickgelu", "")
        if "openai" == argv.model_pretrain
        else argv.model_name
    )
    oc_name = (
        "ViT-SO400M-14-SigLIP-384"
        if "ViT-SO400M-14-SigLIP-378" == oc_name
        else oc_name
    )
    model_torch = open_clip.create_model(
        oc_name, pretrained=argv.model_pretrain
    )
    weights_torch = model_torch.state_dict()
    weights_torch = {
        convert_name(k): v.numpy() for k, v in weights_torch.items()
    }
    weights_torch = transform_weights(
        weights_torch,
        (text_embed, vision_embed, pool_embed),
        (text_heads, vision_heads, pool_heads),
    )

    weights_tf = []
    for w in model_tf.weights:
        assert (
            w.path in weights_torch
        ), f"Can't find weight {w.path} in checkpoint"

        weight = weights_torch.pop(w.path)
        assert (
            w.shape == weight.shape
        ), f"Weight {w.path} shapes not compatible: {w.shape} vs {weight.shape}"

        weights_tf.append(weight)

    if len(weights_torch.keys()):
        raise ValueError(
            f"Some of original weights did not consumed: {weights_torch.keys()}"
        )
    else:
        model_tf.set_weights(weights_tf)

        weights_name = (
            f"{argv.out_path}/{argv.model_name}__"
            f"{argv.model_pretrain}.weights.h5"
        )
        model_tf.save_weights(weights_name)

        weights_hash = hash_file(weights_name)
        os.rename(
            weights_name,
            weights_name.replace(".weights.h5", f"__{weights_hash}.weights.h5"),
        )
