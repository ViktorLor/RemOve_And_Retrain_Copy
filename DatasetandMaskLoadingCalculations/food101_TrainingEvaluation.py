"""
This file is used to evaluate the performance of 101.000 images and applying the 101.000 masks to them, while still getting a good performance.

100 batches with batchsize 64 are loaded and the masks are applied to them.
# Runtime of Tesnor in memory: 43 seconds, 37 seconds
# Rnuntime of Memmap: 39.471134185791016, 38 seconds,
# Runtime of load images from disk


# Time to apply masks for 1 epoch (75200 images):  460 seconds (8 minutes)
"""

# Imports
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
import numpy as np
import PIL as pil
import os
import datetime
import time
import utils

# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
##############################################################################################################
# Create 101.000 masks for the 101.000 images
create_masks = False
memmap = True  # Highly recommended to use memmap, because it is faster and uses less RAM
pt = False  # use pytorch tensors and save one big tensor file which can be loaded with torch.load
singular_files = False  # Save in singular files instead of one big file
##############################################################################################################
# Create random indices for the mask, skipped if already exists
if create_masks and memmap:
	utils.generate_masks(75750, 'food-101_train_masks', 224, memmap=memmap)
	utils.generate_masks(25250, 'food-101_test_masks', 224, memmap=memmap)
if create_masks and pt:
	utils.generate_masks(75750, 'food-101_train_masks', 224)
	utils.generate_masks(25250, 'food-101_test_masks', 224)
if create_masks and singular_files:
	utils.generate_singular_masks(25250, "mask_test", 224)
	utils.generate_singular_masks(75750, "mask_train", 224)

##############################################################################################################
# Load the masks
if memmap == True:
	masks_train = torch.load("random_indices/food-101_train_masks.pt")
	masks_test = torch.load("random_indices/food-101_test_masks.pt")
if pt == True:
	masks_train = np.memmap("random_indices/food-101_train_masks.dat", dtype=np.uint8, mode='r',
	                        shape=(75750, 224, 224))
	masks_test = np.memmap("random_indices/food-101_test_masks.dat", dtype=np.uint8, mode='r', shape=(25250, 224, 224))
if singular_files == True:
	pass
##############################################################################################################
# Load the food101 dataset using split train
start = time.time()
dataset_train = torchvision.datasets.Food101(root='../data', split="train")
dataset_test = torchvision.datasets.Food101(root='../data', split="test")
end = time.time()
print("Time to load data: ", end - start, " seconds")
print("Masks loaded")

##############################################################################################################
"""
Explanation to the dataset logic:

The datase is loaded accordingly to meta/train.json and meta/test.json
The first item is found in train.json and should be a churros.
This is important, because the  masks need to be exactly in the same order as the images.
"""
##############################################################################################################
if False:
	# Memmap
	if memmap or pt:
		dataset_train = utils.DatasetwithMask(dataset_train, masks_train)
	# dataset_test = utils.DatasetwithMask(dataset_test, masks_testr)
	
	# Singular
	if singular_files:
		dataset_train = utils.DatasetSingular(dataset_train, "train")
	
	# load a 64 batch of images and send it to the gpu
	batch_size = 64
	
	loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=0,
	                                     drop_last=True)
	
	mean1 = torch.tensor([0.561]).to(device)
	mean2 = torch.tensor([0.440]).to(device)
	mean3 = torch.tensor([0.312]).to(device)
	start = time.time()
	
	for i, (img, masks, label) in enumerate(loader):
		img = img.to(device)
		label = label.to(device)
		for ii in range(batch_size):
			# mask values=1 should be replaced with [0.561, 0.440, 0.312]
			# mask values=0 should be unchanged
			# masks[ii] >4 : all indices with value 5, 90% of the image
			true_mask = masks[ii].to(device) > 2
			# replace all values in channel 1 with 0.561
			img[ii, 0, true_mask] = mean1
			# replace all values in channel 2 with 0.440
			img[ii, 1, true_mask] = mean2
			# replace all values in channel 3 with 0.312
			img[ii, 2, true_mask] = mean3
		
		# if you want to plot the sum and display images to see if the masks are applied correctly
		# if ii == 0 and i < 3:
		#	plt.imshow(img[ii].permute(1, 2, 0).cpu())
		#	plt.show()
		#	print(torch.sum(true_mask))
		#	print(224 * 224)
		
		if i % 10 == 0:
			print(i)
			print("Time for", i, " batches: ", time.time() - start, " seconds")
	
	end = time.time()
	print("Time to apply masks for 1 epoch: ", (end - start), " seconds")
##############################################################################################################
# Comparison of a normal dataloader epoch
if True:
	dataset_train = torchvision.datasets.Food101(root='../data', split="train", transform=transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.561, 0.440, 0.312], std=[0.252, 0.256, 0.259])
	]))
	
	loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=False, num_workers=0, drop_last=True)
	start = time.time()
	for i, (img, label) in enumerate(loader):
		img = img.to(device)
		label = label.to(device)
		if i % 10 == 0:
			print(i)
			print("Time for", i, " batches: ", time.time() - start, " seconds")
	end = time.time()
	print("Time to apply masks for 1 epoch: ", (end - start), " seconds")
