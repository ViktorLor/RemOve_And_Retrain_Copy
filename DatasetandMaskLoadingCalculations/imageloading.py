"""
Author: Viktor Loreth
Date: 29-07-2023

This file measures the time it takes to load an image from the disk and store it in the gpu memory.

First the image mask is computed using Integrated Gradients. Then the mask is applied to the image.
Now the masked image is saved to the disk. It is measured how long the different methods take.

3 different methods exists. Those are measured and compared:
	- Save the masked image directly to the disk using png. (too high memory usage)
	- Save the attribute map to the disk. Apply the mask to the image in the gpu.
	- Rank the attribute map.

"""

# load the food101 dataset
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

# display variable to show more output
display = False

# Define transforms for training and validation data
transformer = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.561, 0.440, 0.312], std=[0.252, 0.256, 0.259])
])

dataset = torchvision.datasets.Food101(root='../Data', transform=transformer)
# Split the dataset into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
print("Dataset loaded")
# Uncomment to check if the dataset is loaded and displayed correctly
if display:
	# import matplotlib.pyplot as plt
	plt.imshow(train_dataset[0][0].permute(1, 2, 0))
	plt.show()
	
	# get label of image
	label = train_dataset[0][1]
	print("Label: ", label)
	# load from disk the label2name mapping
	with open('../Data/food-101/meta/classes.txt') as f:
		classes = f.readlines()
		classes = [x.strip() for x in classes]
		print("Class: ", classes[label])

size = (3, 224, 244)

########################################################################################################################
# Create random indices for the mask, skipped if already exists
# create 1000 random indices for the mask

if not os.path.exists("random_indices/1000indices.pt"):
	#create dir
	if not os.path.exists("random_indices"):
		os.makedirs("random_indices")
	# create 1000 random masks and save them to a np array dtype=uint8
	masks = torch.zeros((1000, 224, 224), dtype=torch.uint8)
	for x in range(1000):
		# create random mask
		mask_tmp = torch.rand(224, 224).flatten()
		
		indices_onemask = torch.topk(mask_tmp, int(len(mask_tmp)))[1]  # 224 first indices and 203 second indices
		# save the indices
		indices_onemask = torch.stack((indices_onemask // 224, indices_onemask % 224), dim=1)  # 49736x2
		# fill all indices within the first 10,30,50,70,90 with 5,4,3,2,1
		for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
			# get array of indices to fill
			indices_to_fill = indices_onemask[0:int(len(indices_onemask) * threshold)]
			
			masks[x, indices_to_fill[:, 0], indices_to_fill[:, 1]] += 1
	
	torch.save(masks, f"random_indices/1000masks.pt")
else:
	# load the indices from the disk
	masks = torch.load(f"random_indices/1000masks.pt")
print("Masks loaded")
########################################################################################################################
# Show some information about the dataset
print("Dataset: ", len(dataset))
print("Train_dataset: ", len(train_dataset))
print("Val_dataset: ", len(val_dataset))

# store date in logfile
batches=100
with open("results.txt", "w") as f:
	# store today's date and time
	f.write(f"Date: {datetime.datetime.now()}\n")
	f.write(f"Number of batches: {batches}\n")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
########################################################################################################################
skip_method1 = False
if not skip_method1:
	# Method 1: Load the masked images directly to the gpu using png
	
	start = time.time()
	# load a 64 batch of images and send it to the gpu
	loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
	for i, (img, label) in enumerate(loader):
		img = img.to(device)
		label = label.to(device)
		if i == batches:
			break
	stop = time.time()
	print("Method 1: ")
	print("Time to load 100 batches of images(batchsize=64) to gpu using a simple dataloader: ", stop - start, "s")
	########################################################################################################################
	# store result in logfile
	with open("results.txt", "a") as f:
		f.write(
			f"Method 1: Time to load 13 batches of images(batchsize=64) to gpu using a simple dataloader: {stop - start} s\n")

########################################################################################################################
skip_method2 = False
if not skip_method2:
	# Method 2: Apply the indices to the image in the cpu
	start = time.time()
	# load a 64 batch of images and send it to the gpu
	batch_size = 64
	loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
	
	mean1 = torch.tensor([0.561])
	mean2 = torch.tensor([0.440])
	mean3 = torch.tensor([0.312])
	for i, (img, label) in enumerate(loader):
		for ii in range(batch_size):
			# mask values=1 should be replaced with [0.561, 0.440, 0.312]
			# mask values=0 should be unchanged
			# masks[ii] >4 : all indices with value 5, 90% of the image
			true_mask = masks[ii] > 4
			# replace all values in channel 1 with 0.561
			img[ii, 0, true_mask] = mean1
			# replace all values in channel 2 with 0.440
			img[ii, 1, true_mask] = mean2
			# replace all values in channel 3 with 0.312
			img[ii, 2, true_mask] = mean3
		
		img = img.to(device)
		label = label.to(device)
		
		if i == batches:
			break
	stop = time.time()
	print("Method 2: ")
	print("Time to load 100 batches of images(batchsize=64) to gpu using a simple dataloader and applying the mask on the cpu: ",
	      stop - start, "s")
	########################################################################################################################
	# store result in logfile
	with open("results.txt", "a") as f:
		f.write(
			f"Method 2: Time to load 13 batches of images(batchsize=64) to gpu using a simple dataloader and applying the mask: {stop - start} s\n")

########################################################################################################################
skip_method3 = False
if not skip_method3:
	# Method 3: Apply the indices to the image in the gpu
	start = time.time()
	# load a 64 batch of images and send it to the gpu
	batch_size = 64
	loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
	
	mean1 = torch.tensor([0.561]).to(device)
	mean2 = torch.tensor([0.440]).to(device)
	mean3 = torch.tensor([0.312]).to(device)
	for i, (img, label) in enumerate(loader):
		img = img.to(device)
		label = label.to(device)
		for ii in range(batch_size):
			# mask values=1 should be replaced with [0.561, 0.440, 0.312]
			# mask values=0 should be unchanged
			# masks[ii] >4 : all indices with value 5, 90% of the image
			true_mask = (masks[ii] > 4).to(device)
			# replace all values in channel 1 with 0.561
			img[ii, 0, true_mask] = mean1
			# replace all values in channel 2 with 0.440
			img[ii, 1, true_mask] = mean2
			# replace all values in channel 3 with 0.312
			img[ii, 2, true_mask] = mean3
		
		
		if i == batches:
			break
	stop = time.time()
	print("Method 3: ")
	print("Time to load 100 batches of images(batchsize=64) to gpu using a simple dataloader and applying the mask on the gpu: ",
		stop - start, "s")
	########################################################################################################################
	# store result in logfile
	with open("results.txt", "a") as f:
		f.write(f"Method 3: Time to load 13 batches of images(batchsize=64) to gpu using a simple dataloader and "
		        f"applying the mask on the gpu: {stop - start} s\n")
		
		
########################################################################################################################
skip_method4 = False
if not skip_method4:
	# Method 4: Apply the indices to the image in the gpu but calculate the masks using a vectorized approach
	start = time.time()
	# load a 64 batch of images and send it to the gpu
	batch_size = 64
	loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
	
	mean1 = torch.tensor([0.561]).to(device)
	mean2 = torch.tensor([0.440]).to(device)
	mean3 = torch.tensor([0.312]).to(device)
	for i, (img, label) in enumerate(loader):
		img = img.to(device)
		label = label.to(device)
		for ii in range(batch_size):
			# mask values=1 should be replaced with [0.561, 0.440, 0.312]
			# mask values=0 should be unchanged
			# masks[ii] >4 : all indices with value 5, 90% of the image
			true_mask = masks[ii].to(device) > 4
			# replace all values in channel 1 with 0.561
			img[ii, 0, true_mask] = mean1
			# replace all values in channel 2 with 0.440
			img[ii, 1, true_mask] = mean2
			# replace all values in channel 3 with 0.312
			img[ii, 2, true_mask] = mean3
		
		if i == batches:
			break
	stop = time.time()
	print("Method 4: ")
	print(
		"Time to load 100 batches of images(batchsize=64) to gpu using a simple dataloader and applying the mask and the mask generation operation on the gpu: ",
		stop - start, "s")
	########################################################################################################################
	# store result in logfile
	with open("results.txt", "a") as f:
		f.write(f"Method 4: Time to load 13 batches of images(batchsize=64) to gpu using a simple dataloader and "
		        f"applying the mask and the mask generation operation on the gpu: {stop - start} s\n")
