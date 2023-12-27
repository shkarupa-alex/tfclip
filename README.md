# tfclip

Keras (TensorFlow v2) port of **OpenCLIP** package.

+ Based on [Original Pytorch implementation](https://github.com/mlfoundations/open_clip).
+ Contains a lof of models and pretrained weights.
+ Image preprocessing and text tokenization work in graph mode.

## Installation

```bash
pip install tfclip
```

## Differences

1. Function `create_model_and_transforms` returns `model, single_or_batch_image_preprocess, batch_text_preprocess`
   instead of `model, image_preprocess_train, image_preprocess_val` (OpenCLIP).
2. Image preprocessing uses TensorFlow API instead of `PIL` (OpenCLIP). If you want compare ported model outputs
   against original, use pre-resized image.
3. Some weights are too large and can't be uploaded in GitHub release. If you get 404 when trying
   to load model weights, you should convert them locally using script
   `python convert_weights.py <model_name> <pretrain_name> <weights_dir>` and supply weight path like this
   `create_model_and_transforms(..., weights_path='<path_to_weights.h5>')`
4. OpenAI weights moved to `-quickgelu` models where they should be.
5. Model `ViT-SO400M-14-SigLIP-384` renamed to `ViT-SO400M-14-SigLIP-378` (384 // 14 * 14 == 378).

## Examples

Default usage (with pretrained temperature scaling):

```python
import cv2
from keras.src.utils import data_utils
from tfclip import create_model_and_transforms

model, image_prep, text_prep = create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

image = data_utils.get_file(
    'elephant.jpg', 'https://storage.googleapis.com/tensorflow/keras-applications/tests/elephant.jpg')
image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
images = image_prep(image)[None]
texts = text_prep(['a diagram', 'a dog', 'a cat', 'an elephant'])

text_probs = model([images, texts], training=False).numpy()
print('Label probs:', text_probs)  # [[2.3066370e-06 3.2963203e-07 1.9622885e-08 9.9999738e-01]]
# open_clip: [[2.4752687e-06 3.3843190e-07 2.0362965e-08 9.9999714e-01]]
```

Extract image and/or text features separately:

```python
import cv2
import tensorflow as tf
from keras import models
from keras.src.utils import data_utils
from tfclip import create_model_and_transforms

model, image_prep, text_prep = create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

image = data_utils.get_file(
    'elephant.jpg', 'https://storage.googleapis.com/tensorflow/keras-applications/tests/elephant.jpg')
image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
images = image_prep(image)[None]
texts = text_prep(['a diagram', 'a dog', 'a cat', 'an elephant'])

image_model = models.Model(model.inputs[0], model.get_layer('vision/head/out').output)
image_features = image_model(images, training=False)
image_features /= tf.norm(image_features, axis=-1, keepdims=True)

text_model = models.Model(model.inputs[1], model.get_layer('text/head/out').output)
text_features = text_model(texts, training=False)
text_features /= tf.norm(text_features, axis=-1, keepdims=True)

text_probs = tf.matmul(image_features * 100., text_features, transpose_b=True)
text_probs = tf.nn.softmax(text_probs).numpy()
print('Label probs:', text_probs)  # [[2.3066459e-06 3.2963297e-07 1.9622959e-08 9.9999738e-01]]
# open_clip: [[2.4752687e-06 3.3843190e-07 2.0362965e-08 9.9999714e-01]]
```

## Models and weights

> [!TIP]
> Fully ported
> - coca_ViT-B-32: laion2b_s13b_b90k, mscoco_finetuned_laion2b_s13b_b90k
> - coca_ViT-L-14: laion2b_s13b_b90k, mscoco_finetuned_laion2b_s13b_b90k
> - EVA02-B-16: merged2b_s8b_b131k
> - EVA02-L-14: merged2b_s4b_b131k
> - EVA02-L-14-336: merged2b_s6b_b61k
> - ViT-B-16: laion400m_e31, laion400m_e32, laion2b_s34b_b88k, datacomp_xl_s13b_b90k, datacomp_l_s1b_b8k,
    commonpool_l_clip_s1b_b8k, commonpool_l_laion_s1b_b8k, commonpool_l_image_s1b_b8k, commonpool_l_text_s1b_b8k,
    commonpool_l_basic_s1b_b8k, commonpool_l_s1b_b8k
> - ViT-B-16-plus-240: laion400m_e31, laion400m_e32
> - ViT-B-16-quickgelu: openai, metaclip_400m, metaclip_fullcc
> - ViT-B-16-SigLIP: webli
> - ViT-B-16-SigLIP-256: webli
> - ViT-B-16-SigLIP-384: webli
> - ViT-B-16-SigLIP-512: webli
> - ViT-B-16-SigLIP-i18n-256: webli
> - ViT-B-32: laion400m_e31, laion400m_e32, laion2b_e16, laion2b_s34b_b79k, datacomp_xl_s13b_b90k, datacomp_m_s128m_b4k,
    commonpool_m_clip_s128m_b4k, commonpool_m_laion_s128m_b4k, commonpool_m_image_s128m_b4k,
    commonpool_m_text_s128m_b4k, commonpool_m_basic_s128m_b4k, commonpool_m_s128m_b4k, datacomp_s_s13m_b4k,
    commonpool_s_clip_s13m_b4k, commonpool_s_laion_s13m_b4k, commonpool_s_image_s13m_b4k, commonpool_s_text_s13m_b4k,
    commonpool_s_basic_s13m_b4k, commonpool_s_s13m_b4k
> - ViT-B-32-256: datacomp_s34b_b86k
> - ViT-B-32-quickgelu: openai, laion400m_e31, laion400m_e32, metaclip_400m, metaclip_fullcc
> - ViT-L-14: laion400m_e31, laion400m_e32, laion2b_s32b_b82k, datacomp_xl_s13b_b90k, commonpool_xl_clip_s13b_b90k,
    commonpool_xl_laion_s13b_b90k, commonpool_xl_s13b_b90k
> - ViT-L-14-336-quickgelu: openai
> - ViT-L-14-CLIPA: datacomp1b
> - ViT-L-14-CLIPA-336: datacomp1b
> - ViT-L-14-quickgelu: openai, metaclip_400m, metaclip_fullcc

> [!WARNING]
> Local weight conversion required
> - EVA01-g-14: laion400m_s11b_b41k
> - EVA01-g-14-plus: merged2b_s11b_b114k
> - EVA02-E-14: laion2b_s4b_b115k
> - EVA02-E-14-plus: laion2b_s9b_b144k
> - ViT-bigG-14: laion2b_s39b_b160k
> - ViT-bigG-14-CLIPA: datacomp1b
> - ViT-bigG-14-CLIPA-336: datacomp1b
> - ViT-g-14: laion2b_s12b_b42k, laion2b_s34b_b88k
> - ViT-H-14: laion2b_s32b_b79k
> - ViT-H-14-CLIPA: datacomp1b
> - ViT-H-14-CLIPA-336-quickgelu: laion2b, datacomp1b
> - ViT-H-14-quickgelu: metaclip_fullcc
> - ViT-L-16-SigLIP-256: webli
> - ViT-L-16-SigLIP-384: webli
> - ViT-SO400M-14-SigLIP: webli
> - ViT-SO400M-14-SigLIP-378: webli

> [!CAUTION]
> Not ported
> - convnext_base: laion400m_s13b_b51k
> - convnext_base_w: laion2b_s13b_b82k, laion2b_s13b_b82k_augreg, laion_aesthetic_s13b_b82k
> - convnext_base_w_320: laion_aesthetic_s13b_b82k, laion_aesthetic_s13b_b82k_augreg
> - convnext_large_d: laion2b_s26b_b102k_augreg
> - convnext_large_d_320: laion2b_s29b_b131k_ft, laion2b_s29b_b131k_ft_soup
> - convnext_xxlarge: laion2b_s34b_b82k_augreg, laion2b_s34b_b82k_augreg_rewind, laion2b_s34b_b82k_augreg_soup
> - nllb-clip-base: v1
> - nllb-clip-large: v1
> - RN50: openai, yfcc15m, cc12m
> - RN50-quickgelu: openai, yfcc15m, cc12m
> - RN50x4: openai
> - RN50x16: openai
> - RN50x64: openai
> - RN101: openai, yfcc15m
> - RN101-quickgelu: openai, yfcc15m
> - roberta-ViT-B-32: laion2b_s12b_b32k
> - xlm-roberta-base-ViT-B-32: laion5b_s13b_b90k
> - xlm-roberta-large-ViT-H-14: frozen_laion5b_s13b_b90k

## Citation

```
@software{ilharco_gabriel_2021_5143773,
  author       = {Ilharco, Gabriel and
                  Wortsman, Mitchell and
                  Wightman, Ross and
                  Gordon, Cade and
                  Carlini, Nicholas and
                  Taori, Rohan and
                  Dave, Achal and
                  Shankar, Vaishaal and
                  Namkoong, Hongseok and
                  Miller, John and
                  Hajishirzi, Hannaneh and
                  Farhadi, Ali and
                  Schmidt, Ludwig},
  title        = {OpenCLIP},
  month        = jul,
  year         = 2021,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5143773},
  url          = {https://doi.org/10.5281/zenodo.5143773}
}
```

```
@inproceedings{Radford2021LearningTV,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and A. Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={ICML},
  year={2021}
}
```

```
@inproceedings{schuhmann2022laionb,
  title={{LAION}-5B: An open large-scale dataset for training next generation image-text models},
  author={Christoph Schuhmann and
          Romain Beaumont and
          Richard Vencu and
          Cade W Gordon and
          Ross Wightman and
          Mehdi Cherti and
          Theo Coombes and
          Aarush Katta and
          Clayton Mullis and
          Mitchell Wortsman and
          Patrick Schramowski and
          Srivatsa R Kundurthy and
          Katherine Crowson and
          Ludwig Schmidt and
          Robert Kaczmarczyk and
          Jenia Jitsev},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  url={https://openreview.net/forum?id=M3Y74vmsMcY}
}
```

```
@inproceedings{Yu2022CoCaCC,
  title   = {CoCa: Contrastive Captioners are Image-Text Foundation Models},
  author  = {Jiahui Yu and Zirui Wang and Vijay Vasudevan and Legg Yeung and Mojtaba Seyedhosseini and Yonghui Wu},
  year    = {2022}
}
```

```
@article{datacomp,
  title={DataComp: In search of the next generation of multimodal datasets},
  author={Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, Eyal Orgad, Rahim Entezari, Giannis Daras, Sarah Pratt, Vivek Ramanujan, Yonatan Bitton, Kalyani Marathe, Stephen Mussmann, Richard Vencu, Mehdi Cherti, Ranjay Krishna, Pang Wei Koh, Olga Saukh, Alexander Ratner, Shuran Song, Hannaneh Hajishirzi, Ali Farhadi, Romain Beaumont, Sewoong Oh, Alex Dimakis, Jenia Jitsev, Yair Carmon, Vaishaal Shankar, Ludwig Schmidt},
  journal={arXiv preprint arXiv:2304.14108},
  year={2023}
}
```

```
@inproceedings{cherti2023reproducible,
  title={Reproducible scaling laws for contrastive language-image learning},
  author={Cherti, Mehdi and Beaumont, Romain and Wightman, Ross and Wortsman, Mitchell and Ilharco, Gabriel and Gordon, Cade and Schuhmann, Christoph and Schmidt, Ludwig and Jitsev, Jenia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2818--2829},
  year={2023}
}
```

```
@inproceedings{li2023clipa,
      title={An Inverse Scaling Law for CLIP Training},
      author={Xianhang Li and Zeyu Wang and Cihang Xie},
      booktitle={NeurIPS},
      year={2023},
}
```

```
@article{li2023clipav2,
      title={CLIPA-v2: Scaling CLIP Training with 81.1% Zero-shot ImageNet Accuracy within a $10,000 Budget; An Extra $4,000 Unlocks 81.8% Accuracy},
      author={Xianhang Li and Zeyu Wang and Cihang Xie},
      journal={arXiv preprint arXiv:2306.15658},
      year={2023},
}
```

```
@article{EVA-CLIP,
  title={EVA-CLIP: Improved Training Techniques for CLIP at Scale},
  author={Sun, Quan and Fang, Yuxin and Wu, Ledell and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2303.15389},
  year={2023}
}
```

```
@inproceedings{xu2023metaclip,
   title={Demystifying CLIP Data},
   author={Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu, Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer and Christoph Feichtenhofer},
   journal={arXiv preprint arXiv:2309.16671},
   year={2023}
}
```

```
@article{zhai2023sigmoid,
  title={Sigmoid loss for language image pre-training},
  author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
  journal={arXiv preprint arXiv:2303.15343},
  year={2023}
}
```

```
@article{fang2023data,
  title={Data Filtering Networks},
  author={Fang, Alex and Jose, Albin Madappally and Jain, Amit and Schmidt, Ludwig and Toshev, Alexander and Shankar, Vaishaal},
  journal={arXiv preprint arXiv:2309.17425},
  year={2023}
}
```