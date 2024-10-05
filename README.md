# Sum-of-Parts (SOP) Models: Faithful Attributions for Groups of Features

\[[Paper](https://arxiv.org/abs/2310.16316)\] \[[Blog](https://debugml.github.io/sum-of-parts/)\]

Official implementation for "Sum-of-Parts Models: Faithful Attributions for Groups of Features".

Authors: Weiqiu You, Helen Qu, Marco Gatti, Bhuvnesh Jain, Eric Wong

## TODO
- [x] Release updated code - Oct 2nd 2024
- [x] Update arxiv - Oct 5th 2024

## Prerequisite

To set up the environment:

```
conda create -n sop python=3.10
conda activate sop
pip install -r requirements.txt
```

To do experiments on ImageNet first 10 classes, create a folder `data/imagenet_m` with subfolders `data/imagenet_m/train` and `data/imagenet_m/val`, download data from ImageNet and put the 10 classes of data in subfolders in these folders.

## Usage


