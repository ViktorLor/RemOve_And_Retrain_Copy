# Interpretable AI 

To use this notebook you need to have the following installed:
- numpy
- pytorch
- torchvision
- matplotlib

Download some images to test saliency maps. I personally used following images:
- https://github.com/EliSchwartz/imagenet-sample-images

# Papers used in this notebook:
A Benchmark for Interpretability Methods in Deep Neural Networks (https://arxiv.org/abs/1806.10758)
Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)

# Code used and copied from the following sources:
Extracting the ImageNetData https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh


# Roar Implementation

The implementation varies from the original paper as I am using pytorch and captum instead of tensorflow and saliency.

## Architecture:

The architecture was chosen to be the ResNet18 model because it is easier.

- Model: ResNet18, https://arxiv.org/abs/1512.03385
- Dataset: ImageNet, http://www.image-net.org/ 

Roar constists of 3 steps:

1. Train a classification model on original dataset. -> Done using the ResNET 18 weights.
2. Use the trained model to extract attribution maps for each image.
3. Retrain the model on the original dataset with the attribution maps as additional features. 

## TODO from a code perspective

- Extract the attribution maps for each image. Integrated Gradients, Gradient Saliency, Guide Backpropagation
- Retrain the model on the original dataset with the attribution maps blanked out as additional features.
- Calculate the accuracy of the model on the retrained model.

