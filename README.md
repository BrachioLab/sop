# Sum-of-Parts (SOP) Models: Faithful Attributions for Groups of Features

\[[Paper](https://arxiv.org/abs/2310.16316)\] \[[Blog](https://debugml.github.io/sum-of-parts/)\]

Official implementation for "Sum-of-Parts: Faithful Attributions for Groups of Features".

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

Also need to install `dev` branch of [`exlib`](https://github.com/BrachioLab/exlib/tree/dev)

## Usage

### Demos
Here we show how to use pretrained SOP and train your own SOP models for ImageNet and CosmoGrid
1. [ImageNet](https://github.com/BrachioLab/sop/blob/main/notebooks/demos/imagenet.ipynb)
2. [CosmoGrid](https://github.com/BrachioLab/sop/blob/main/notebooks/demos/cosmogrid.ipynb)

### Evaluation
Here we show how we evaluate. The actual scripts we run are in `src/sop/run`.
1. [ImageNet Accuracy](https://github.com/BrachioLab/sop/blob/main/notebooks/metrics/imagenet_s_acc_purity.ipynb)
2. [ImageNet Purity](https://github.com/BrachioLab/sop/blob/main/notebooks/metrics/imagenet_s_purity.ipynb)
3. [ImageNet Insertion Deletion](https://github.com/BrachioLab/sop/blob/main/notebooks/metrics/ins_del_mod.ipynb)
4. [ImageNet Sparsity](https://github.com/BrachioLab/sop/blob/main/notebooks/metrics/sparsity.ipynb)
5. [ImageNet Fidelity](https://github.com/BrachioLab/sop/blob/main/notebooks/metrics/fidelity.ipynb)
6. [Cosmogrid Accuracy and Purity](https://github.com/BrachioLab/sop/blob/main/notebooks/metrics/cosmogrid_acc_purity.ipynb)

## Citation
```
@misc{you2024sumofpartsfaithfulattributionsgroups,
      title={Sum-of-Parts: Faithful Attributions for Groups of Features}, 
      author={Weiqiu You and Helen Qu and Marco Gatti and Bhuvnesh Jain and Eric Wong},
      year={2024},
      eprint={2310.16316},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2310.16316}, 
}
```
