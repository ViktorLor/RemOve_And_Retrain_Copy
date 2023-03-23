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


# Roar Implementation

The implementation varies from the original paper as I am using pytorch and captum instead of tensorflow and saliency.

## Architecture:

The architecture was chosen to be the ResNet50 model because it is easier.

- Model: ResNet50, https://arxiv.org/abs/1512.03385
- Dataset: Food101

Roar constists of 3 steps:

1. Train a classification model on original dataset. -> Done using the ResNET 50 weights.
2. Use the trained model to extract attribution maps for each image.
3. Retrain the model on the original dataset with the attribution maps as additional features. 

## Model Parameters for Food 101:
    food_101_params = {
        'train_batch_size': 256,
        'num_train_images': 75750,
        'num_eval_images': 25250,
        'num_label_classes': 101,
        'num_train_steps': 20000,
        'base_learning_rate': 0.7,
        'weight_decay': 0.0001,
        'eval_batch_size': 256,
        'mean_rgb': [0.561, 0.440, 0.312],
        'stddev_rgb': [0.252, 0.256, 0.259]
    }
