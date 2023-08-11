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
if param1 == 'generate_random_masks':
	
	if not os.path.exists(path + 'indices_to_block/random_baseline/'):
		os.makedirs(path + 'indices_to_block/random_baseline/')
	
	utils.generate_random_masks_3D(101000, path, 'indices_to_block/random_baseline/', images_to_copy='images',
	                               image_size=224, saveaspng=True)

if param1 == 'integrated_gradient':
	
	if not os.path.exists(path + 'indices_to_block/integrated_gradient/'):
		os.makedirs(path + 'indices_to_block/integrated_gradient/')
	
	model = torchvision.models.resnet50(weights=True)  # Need to adjust for food101
	model.eval()
	utils.generate_saliency_masks_3D(model, 'integrated_gradient', path, 224, True, saveaspng=True)
	
	
	"""
		for z, folder in enumerate(folders):
		images = os.listdir(path + '/' + folder)
		for i in range(len(thresholds)):
			if not os.path.exists(path + '/' + folder + str(int(thresholds[i] * 100))):
				os.makedirs(path + '/' + folder + str(int(thresholds[i] * 100)))
		
		for i, image in enumerate(images):
			try:
				sal_help.calculate_saliency_map(model, image, thresholds=thresholds, cuda=use_cuda,
				                                project_path=path + '/' + folder)
			except:
				print("Error with image: ", image)
				# write to logfile
				with open("logfile.txt", "a") as logfile:
					logfile.write("Error with image: " + image)
			
			if i % 100 == 0:
				print(i, z, "image done", len(folders))
	"""

# %%
