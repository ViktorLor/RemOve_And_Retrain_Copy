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

folder = 'random_baseline'

food101path = '../Data/food-101'
# threshold: threshold: >4.5: 10%, 3.5: 30%, 2.5: 50%, 1.5: 70%, 0.5: 90%,masked
threshold_to_string = {4.5: "10", 3.5: "30", 2.5: "50", 1.5: "70", 0.5: "90"}
for threshold in [4.5, 3.5, 2.5, 1.5, 0.5]:
	for i in range(5):
		# create dir
		if not os.path.exists('../models/food101/' + folder + '/' + threshold_to_string[threshold]):
			os.makedirs('../models/food101/' + folder + '/' + threshold_to_string[threshold])
		
		train_dataset = utils.Food101MaskDataset(data_folder_images=food101path + '/images/',
		                                         data_folder_mask=food101path + '/indices_to_block/' + folder + '/',
		                                         meta_file=food101path + '/meta/train.txt',
		                                         threshold=threshold, transform=transformer)
		
		print("Train Dataset: ", len(train_dataset))
		print("Size should be: ", 75750)
		
		utils.training_food101(train_dataset, folder + '/' + threshold_to_string[threshold] + f'/Resnet_50_run_{i}',
		                       device,
		                       shuffle=True, seed=i)
	
	del train_dataset


for threshold in [4.5, 3.5, 2.5, 1.5, 0.5]:
	for i in range(5):
		
		# Load the trained model:
		model = models.resnet50()
		# Replace the last layer with a new fully connected layer
		num_ftrs = model.fc.in_features
		model.fc = nn.Linear(num_ftrs, 101)
		
		model.load_state_dict(torch.load('../models/food101/10_ResNet50_Food101.pth'))
		model.eval()
		
		test_dataset = utils.Food101MaskDataset(data_folder_images=food101path + '/images/',
		                                        data_folder_mask=food101path + '/indices_to_block/random_baseline/',
		                                        meta_file=food101path + '/meta/test.txt',
		                                        threshold=threshold, transform=transformer)
		
		print("Test Dataset: ", len(test_dataset))
		print("Size should be: ", 25250)
		
		utils.test_food101(test_dataset, model, device, result_file='10_ResNet50_food101_results.txt', shuffle=False, seed=0)
