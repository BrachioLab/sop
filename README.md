# Sum-of-Parts (SOP) Models: Faithful Attributions for Groups of Features

\[[Paper](https://arxiv.org/abs/2310.16316)\] \[[Blog](https://debugml.github.io/sum-of-parts/)\]

Official implementation for "Sum-of-Parts Models: Faithful Attributions for Groups of Features".

Authors: Weiqiu You, Helen Qu, Marco Gatti, Bhuvnesh Jain, Eric Wong

## Prerequisite

To set up the environment:

```
conda create -n sop python=3.10
conda activate sop
pip install -r requirements.txt
```

To do experiments on ImageNet first 10 classes, create a folder `data/imagenet_m` with subfolders `data/imagenet_m/train` and `data/imagenet_m/val`, download data from ImageNet and put the 10 classes of data in subfolders in these folders.

## Usage

### Training

To train SOP for 10 classes on ImageNet on the Huggingface's Vision Transformer `google/vit-base-patch16-224`, first download our [model](https://drive.google.com/file/d/1WDUSvGtBwyGq5PYFke6HvR8fWK8NghQb/view?usp=drive_link) for the first 10 classes for ImageNet.

```
python scripts/run/train_imagenet_m.py
```

or notebook `notebooks/train.ipynb`.

### Evaluation

To use the trained SOP wrapped model at inference time, checkout `notebooks/eval.ipynb`.