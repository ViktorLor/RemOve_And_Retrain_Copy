# RemOve_And_Retrain_Copy

This program is based on the original paper of:
A Benchmark for Interpretability Methods in Deep Neural Networks (https://arxiv.org/abs/1806.10758)

To use this notebook you need to have the following installed:
- numpy
- pytorch
- torchvision
- matplotlib

Download some images to test saliency maps. I personally used following images:
- https://github.com/EliSchwartz/imagenet-sample-images

# Roar Implementation

The implementation varies from the original paper as I am using pytorch and captum instead of tensorflow and saliency.

## Architecture:

The architecture was chosen to be the ResNet50 model because it is easier.

- Model: ResNet50, https://arxiv.org/abs/1512.03385
- Dataset: Food101

Roar consists of 3 steps:

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


The project is saved in following structure:

1. archive
    - contains deprecated code and playgrounds
2. data
    - contains the datasets for the project
    - the datasets are not included in the repository, they need to be downloaded
    - the altered datasets are saved in the respective folder
    - data/mnist
      - mod_imagestest: modified mnist test dataset. saved with 0.1, 0,3, 0.5, 0.7, 0.9
      - mod_imagestrain: modified mnist training dataset
      - mod_randomtest: randomly modified mnist test dataset
      - mod_randomtrain: randomly modified mnist training dataset
      - raw: original data downloaded
3. DatasetandMaskLoadingCalculatinons
    - Contains scripts to load the datasets and applying the mask on the fly and calculations
4. Feedback
    - contains the feedback for the project report
5. Latex
    - contains the latex files for the project report
6. models
    - contains the models and the respective accuracy and loss plots
7. Papers
    - contains the papers used for the project
8. Reporting
   - contains additional scripts to generate plots and accuracy tables
9. Saliency
    - Is used to create the saliency blocked images for the datasets -> Old Approach which saves all images 
10. Training
    - contains the training scripts for the models


# Using the project to generate the models:

1. Train the ResNET50 model on the original dataset. Training/Train_{dataset}_{model}_original_Dataset
2. Check the generated accuracy in models/{dataset}/original_accuracy.png
3. Generate the saliency maps for the dataset. Saliency/saliency_{dataset}.
4. Check if the files are generated correctly in data/{dataset}/mod_images{train_test}/{0-0.9}
5. Train the model on the modified dataset. Training/Train_{dataset}_{model}_5times_thresholdblock
6. Check the generated accuracy in models/{dataset}/{method}/{0.1-0.9}_accuracy.txt
7. Generate Plots and Reports as needed in Reporting
