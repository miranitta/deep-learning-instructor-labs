# Common Student Questions

## Why do CNNs work well on images?
Because convolutions learn local spatial patterns and reuse the same filters across the image.

## Why do we use pooling?
Pooling reduces spatial size, lowers compute cost, and adds some translation robustness.

## Why freeze pretrained layers?
To preserve useful learned features and reduce the number of trainable parameters on small datasets.

## What causes overfitting?
Too much model capacity, too little data, weak regularization, or too many epochs.
