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
food101path = '../data/food-101'
train_dataset = utils.Food101MaskDataset(data_folder_images=food101path + '/images',
                                         data_folder_mask=food101path + '/indices_to_block/random_baseline',
                                         threshold=5, transform=transformer)

train_dataset[0]

print("Loaded Dataset")
print("Train Dataset: ", len(train_dataset))
