"""
Training the ResNet-50 model on the Food-101 dataset.

Parameters used: https://github.com/google-research/google-research/blob/master/interpretability_benchmark/train_resnet.py#L113

30h Training by now.
Probably possible to crease batch size to increase speed; -> But then paper is not completely reproducible
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torchvision import models
import os
import time
import utils

# define seed
torch.manual_seed(0)

# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
# Define transforms for training and validation data
transformer = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.561, 0.440, 0.312], std=[0.252, 0.256, 0.259])
])

# Load the Food 101 dataset train and test with the original split of the dataset
train_dataset = torchvision.datasets.Food101(root='../Data', transform=transformer, download=True, split="train")

print("Loaded Dataset")
print("Train Dataset: ", len(train_dataset))
print("Size should be: ", 75750)

# Fully builds a trained model and saves it.
utils.training_food101(train_dataset, 'original_ResNet50', device, shuffle=True, seed=0)

del train_dataset
# Load the trained model:
model = models.resnet50()
# Replace the last layer with a new fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 101)

model.load_state_dict(torch.load('../models/food101/original_ResNet50.pth'))
model.eval()

test_dataset = torchvision.datasets.Food101(root='../Data', transform=transformer, download=True, split="test")
print("Test Dataset: ", len(test_dataset))
print("Size should be: ", 25250)

utils.test_food101(test_dataset, model, device, result_file='original_results.txt', shuffle=False, seed=0)
