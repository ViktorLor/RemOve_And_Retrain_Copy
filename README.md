# RemOve_And_Retrain Library 

This repository implements the ROAR method, based on the paper:
"A Benchmark for Interpretability Methods in Deep Neural Networks"
https://arxiv.org/abs/1806.10758

# Requirements

Install the following dependencies before running the notebooks:

numpy
torch
torchvision
matplotlib

To test saliency maps, download example images such as:
https://github.com/EliSchwartz/imagenet-sample-images

# Roar Implementation

This implementation reproduces the ROAR approach using PyTorch and Captum instead of TensorFlow and Saliency.

## Architecture:

The architecture was chosen to be the ResNet50 model because it is easy to train.

- Model: ResNet50, https://arxiv.org/abs/1512.03385
- Dataset: Food101, MNIST Numbers

## ROAR Workflow

1. Train a classification model on the original dataset.
2. Generate attribution maps (saliency maps) for each image using the trained model.
3. Retrain the model on datasets where the most important pixels (according to the attribution maps) have been removed.

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


# Project Structure

```graphql
archive/                      # Deprecated code and experiments
data/                         # Dataset folder
  └── mnist/
      ├── mod_imagestest/     # Modified MNIST test datasets (0.1–0.9 thresholds)
      ├── mod_imagestrain/    # Modified MNIST training datasets
      ├── mod_randomtest/     # Randomly modified test datasets
      ├── mod_randomtrain/    # Randomly modified training datasets
      └── raw/                # Original dataset
DatasetandMaskLoadingCalculations/  # Scripts for dataset loading and masking
Latex/                        # LaTeX report files
models/                       # Trained models and accuracy/loss plots
Papers/                       # Reference papers
Reporting/                    # Plot and report generation scripts
Saliency/                     # Saliency map generation 
Training/                     # Model training scripts
```


# Using the project to generate the models:

Workflow: Generating and Evaluating Models

1. Train the base ResNet50 model. This is required to have the original model and create the original attribution maps.
 ```bash
Training/Train_{dataset}_{model}_original_Dataset
 ```
2. Evaluate and verify accuracy:
```bash
models/{dataset}/original_accuracy.txt
 ```

3. Build the saliency image data for the dataset.
```bash
Saliency/saliency_{dataset}
```
4. Control if the modified dataset are correct
```bash
data/{dataset}/mod_images{train,test}/{0.1–0.9}
```
5. Retrain the dataset 5 times on the various thresholds.
```bash
Training/Train_{dataset}_{model}_5times_thresholdblock
```

6. Check the performance Metrics
```bash
models/{dataset}/{method}/{0.1–0.9}_accuracy.txt
```

7. Generate plots and reports:
```bash
Use scripts in Reporting/
```