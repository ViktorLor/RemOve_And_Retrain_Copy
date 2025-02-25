"""
Created on Mon Nov  4 14:50:00 2019

Author: Viktor Loreth

Generate random masks for the food101 dataset. Those are saved as a memmap file.
"""
import torchvision

import sys
import os
import torch
import numpy as np
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.nn as nn
from PIL import Image
import utils as utils

# LINUX PATH
path = r'/home/viktorl/Intepretable_AI_PR_Loreth/Data/food-101/'

# WINDOWS PATH
#path = f'C:\\Users\\Vik\\Documents\\4. Private\\01. University\\2023_Sem6\\Intepretable_AI\\data\\food-101\\'

if not os.path.exists(path + 'indices_to_block/'):
	os.makedirs(path + '/indices_to_block/', exist_ok=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.cuda.empty_cache()
torch.cuda.synchronize()

# Load first parameter "generate_random_masks"
param1 = sys.argv[1]
print(param1)
if param1 == 'random_baseline':
	
	if not os.path.exists(path + 'indices_to_block/random_baseline/'):
		os.makedirs(path + 'indices_to_block/random_baseline/')
	
	utils.generate_random_masks_3D(101000, path, 'indices_to_block/random_baseline/', images_to_copy='images',
	                               image_size=224, saveaspng=True)

if param1 == 'integrated_gradient':
	
	if not os.path.exists(path + 'indices_to_block/integrated_gradient/'):
		os.makedirs(path + 'indices_to_block/integrated_gradient/')
	
	# load weights
	model = torchvision.models.resnet50()
	# Replace the last layer with a new fully connected layer
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, 101)
	# load weights from file
	model.load_state_dict(torch.load(r'/home/viktorl/Intepretable_AI_PR_Loreth/models/food101/runs_original/original_ResNet50_lr_0_7_0.pth'))
	
	model.eval()
	utils.generate_saliency_masks_3D(model, 'integrated_gradient', path, 224, test=False, saveaspng=True)

# %%

if param1 == 'guided_backprop':
	
	if not os.path.exists(path + 'indices_to_block/guided_backprop/'):
		os.makedirs(path + 'indices_to_block/guided_backprop/')
	
	# load weights
	model = torchvision.models.resnet50()
	# Replace the last layer with a new fully connected layer
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, 101)
	# load weights from file
	model.load_state_dict(torch.load(
		r'/home/viktorl/Intepretable_AI_PR_Loreth/models/food101/runs_original/original_ResNet50_lr_0_7_0.pth'))
	
	model.eval()
	utils.generate_saliency_masks_3D(model, 'guided_backprop', path, 224, test=False, saveaspng=True)

# %%

