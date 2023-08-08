"""
Utils file for custom training using masks and threshold blocks.

The added functions are a custom Dataset and a custom DataLoader.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torchvision import models
import os
import time
import numpy as np
from PIL import Image


# Mask Dataset which loads data from Food101 and additionally loads the masks and applies them to the images
class Food101MaskDataset(torch.utils.data.Dataset):
	def __init__(self, data_folder_images, data_folder_mask, threshold, transform=None):
		# threshold: 4: 10%, 3: 30%, 2: 50%, 1: 70%, 0: 90%,masked
		try:
			self.threshold = threshold
			self.data_folder_images = data_folder_images
			self.data_folder_mask = data_folder_mask
			self.classes = os.listdir(data_folder_images)  # Get class names from subdirectories
			self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
			self.transform = transform
			
			# Create a list of (image_path,mask_path, label) pairs
			self.samples = []
			
			for idx, cls in enumerate(self.classes):
				class_folder_images = os.path.join(data_folder_images, cls)
				for img_filename in os.listdir(class_folder_images):
					img_path = os.path.join(class_folder_images, img_filename)
					mask_path = os.path.join(data_folder_mask, cls, img_filename[:-4] + '.png')
					self.samples.append((img_path, mask_path, idx))
		except Exception as e:
			print(self.samples)
			print("Error loading dataset: ", e)
			print("Please abort the program")
			quit(1)
	
	def __len__(self):
		return len(self.samples)
	
	def __getitem__(self, idx):
		img_path, mask_path, label = self.samples[idx]
		
		image = Image.open(img_path).convert('RGB')
		mask = Image.open(mask_path).convert('RGB')
		
		mask = torch.Tensor(np.array(mask))
		# convert to 3x224x224
		mask = mask.permute(2, 0, 1)
		
		if self.transform:
			image = self.transform(image)
		
		mean_1, mean_2, mean_3 = 0.485, 0.456, 0.406
		
		# Apply mask to image
		image[0, :, :] = torch.where(mask[0, :, :] > self.threshold, image[0, :, :], torch.tensor(mean_1))
		image[1, :, :] = torch.where(mask[1, :, :] > self.threshold, image[1, :, :], torch.tensor(mean_2))
		image[2, :, :] = torch.where(mask[2, :, :] > self.threshold, image[2, :, :], torch.tensor(mean_3))
		
		#plot image
		import matplotlib.pyplot as plt
		plt.imshow(image.permute(1,2,0))
		plt.show()
		
		return image, label
