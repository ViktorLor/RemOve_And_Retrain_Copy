"""
Created on Mon Nov  4 14:50:00 2019

Author: Viktor Loreth

This file is used to create a random baseline for the saliency maps. The output are images with a random mask applied
which masks x% of the image. The images are saved in the folder ../data/MNIST/randombaseline_imagestest and
../data/MNIST/randombaseline_imagestrain.

"""
import sys
import torchvision
import os
import torch
from torchvision import transforms
from captum.attr import IntegratedGradients

# load model from ../Training/Train_Mnist.py
sys.path.append("C:\\Users\\Vik\\Documents\\4. Private\\01. University\\2023_Sem6\\Intepretable_AI\\Training")
from Train_MNIST_SimpleCNN_original_Dataset import Net

import numpy as np
import time
import random

# torch seed
torch.manual_seed(0)

# Windows path:
pathtrain = 'C:\\Users\Vik\Documents\\4. Private\\01. University\\2023_Sem6\\Intepretable_AI\\data\\MNIST\\randombaseline_imagestrain'
pathtest = 'C:\\Users\Vik\Documents\\4. Private\\01. University\\2023_Sem6\\Intepretable_AI\\data\\MNIST\\randombaseline_imagestest'

for path in [pathtrain, pathtest]:
	
	thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
	
	# Define the transform to apply to the input images
	transform = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize((0.1307,), (0.3081,))])
	
	# Load the training and testing data
	if path == pathtrain:
		trainset = torchvision.datasets.MNIST(root='../data', train=True,
		                                      download=True, transform=transform)
	else:
		trainset = torchvision.datasets.MNIST(root='../data', train=False,
		                                      download=True, transform=transform)
	
	print("Data loaded")
	# generate a folder for each threshold
	if not os.path.exists(path + '\\0.3'):
		# add a folder for the original images
		os.mkdir(path + '\\0')
		# generate a folder for each class
		for i in range(10):
			os.mkdir(path + f'\\0\\{i}')
		
		for threshold in thresholds:
			os.mkdir(path + f'\\{threshold}')
			# generate a folder for each class
			for i in range(10):
				os.mkdir(path + f'\\{threshold}\\{i}')
	else:
		exit(1)
	print("Folders created")
	# open a logfile to save the labels
	with open(path + '\\labels.txt', 'w') as f:
		# iterate over all images in the trainset
		for i in range(len(trainset)):
			# start timer
			start = time.time()
			# get the image an the label
			img, label = trainset[i]
			# send img to device
			# save the label in the logfile and end the line
			f.write(f'{i}, {label} \n')
			
			
			for ii in range(len(thresholds)):
				img_tmp = img.clone()
				img_masked = torch.zeros(28, 28)
				all_indices = [(i, j) for i in range(28) for j in range(28)]
				indices = random.sample(all_indices, k=int(thresholds[ii] * len(all_indices)))
				# convert indices to torch
				indices = torch.tensor(indices)
				# use indices to mask the img
				img_masked[indices[:, 0], indices[:, 1]] = 1
				# print
				
				# set the pixels of the image to mean value of MNIST numbers
				img_tmp = img_tmp * (1 - img_masked) + img_masked * 0.1307
				
				# save the image
				torchvision.utils.save_image(img_tmp, path + f'\\{thresholds[ii]}\\{label}\\{i}.png')
			
			# track progress
			if i % 1000 == 0 and i != 0:
				print(f'{i} images processed')
				# print timer progress
				print(f'{time.time() - start} seconds per image')
	print("success")
