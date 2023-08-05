"""
Utils file for creating masks and computing mask operations;
"""
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_masks(dataset_size, name, image_size=224, memmap=False, thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]):
	"""
	
	:param dataset_size: how many masks to create
	:param image_size: which shape the masks should have [224,224]
	:param thresholds: which thresholds to use
	:return:
	
	if huge = True the memory is bigger than the RAM. Then the masks are saved in a memmap file.
	"""
	
	if memmap:
		masks_memmap = np.memmap(f"random_indices/{name}.dat", dtype=np.uint8, mode='w+',
		                         shape=(dataset_size, image_size, image_size))
		
		for x in range(dataset_size):
			mask_tmp = np.random.rand(image_size, image_size).flatten()
			
			indices_onemask = np.argsort(mask_tmp)[::-1]  # reverse the array
			# save the indices
			indices_onemask = np.stack((indices_onemask // image_size, indices_onemask % image_size), axis=1)
			
			# fill all indices within the first 10,30,50,70,90 with 5,4,3,2,1
			for threshold in thresholds:
				# get array of indices to fill
				indices_to_fill = indices_onemask[0:int(len(indices_onemask) * threshold)]
				
				masks_memmap[x, indices_to_fill[:, 0], indices_to_fill[:, 1]] += 1
			
			if x % 1000 == 0:
				print(f"Generated {x} masks")
	
	
	else:
		masks = torch.zeros((dataset_size, image_size, image_size), dtype=torch.uint8)
		
		for x in range(dataset_size):
			mask_tmp = torch.rand(224, 224).flatten()
			
			indices_onemask = torch.topk(mask_tmp, int(len(mask_tmp)))[1]  # 224 first indices and 203 second indices
			# save the indices
			indices_onemask = torch.stack((indices_onemask // 224, indices_onemask % 224), dim=1)  # 49736x2
			
			# fill all indices within the first 10,30,50,70,90 with 5,4,3,2,1
			for threshold in thresholds:
				# get array of indices to fill
				indices_to_fill = indices_onemask[0:int(len(indices_onemask) * threshold)]
				
				masks[x, indices_to_fill[:, 0], indices_to_fill[:, 1]] += 1
			
			if x % 1000 == 0:
				print(f"Generated {x} masks")
		
		torch.save(masks, f"random_indices/{name}.pt")
		print(f"Saved {dataset_size} masks")
	
	print("Masks created")


def generate_singular_masks(dataset_size, name, image_size=224, thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]):
	"""
	This is purely for testing purposes. It generates masks with names 0 to dataset_size.
	:param dataset_size:
	:param name:
	:param image_size:
	:param thresholds:
	:return:
	"""
	
	for x in range(dataset_size):
		mask = torch.zeros((image_size, image_size), dtype=torch.uint8)
		
		mask_tmp = torch.rand(224, 224).flatten()
		
		indices_onemask = torch.topk(mask_tmp, int(len(mask_tmp)))[1]  # 224 first indices and 203 second indices
		# save the indices
		indices_onemask = torch.stack((indices_onemask // 224, indices_onemask % 224), dim=1)  # 49736x2
		
		# fill all indices within the first 10,30,50,70,90 with 5,4,3,2,1
		for threshold in thresholds:
			# get array of indices to fill
			indices_to_fill = indices_onemask[0:int(len(indices_onemask) * threshold)]
			
			mask[indices_to_fill[:, 0], indices_to_fill[:, 1]] += 1
		
		torch.save(mask, f"{name}/{x}.pt")
		
		if x % 1000 == 0:
			print(f"Generated {x} masks")


class DatasetwithMask(torch.utils.data.Dataset):
	def __init__(self, dataset, masks):
		self.dataset = dataset
		self.masks = masks
		self.transform = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.561, 0.440, 0.312], std=[0.252, 0.256, 0.259])
		])
	
	def __len__(self):
		return len(self.dataset)
	
	def __getitem__(self, idx):
		image, label = self.dataset[idx]
		mask = self.masks[idx]
		
		if self.transform:
			image = self.transform(image)
		
		return image, mask, label


class DatasetSingular(torch.utils.data.Dataset):
	def __init__(self, dataset, flag):
		self.dataset = dataset
		self.transform = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.561, 0.440, 0.312], std=[0.252, 0.256, 0.259])
		])
		self.flag = flag
	def __len__(self):
		return len(self.dataset)
	
	def __getitem__(self, idx):
		image, label = self.dataset[idx]

		mask = torch.load(f"mask_{self.flag}/{idx}.pt")
		
		if self.transform:
			image = self.transform(image)
		
		return image, mask, label