# Cosmogrid

## Segmentation

To use segments instead of patches, we first need to generate the segments:

```
seg_cosmogrid.ipynb
```

## Train

Given an existing CNN model, to train SOP on top of the frozen model, use

```
train_cosmogrid.ipynb
```

## Eval

To get the groups from a trained SOP model and save them, we use

```
eval_cosmogrid.ipynb
```

## Analyze

To analyze voids and clusters (and you can add other cosmological structures that you would like to analyze), we use

```
analyze_cosmogrid.ipynb
```