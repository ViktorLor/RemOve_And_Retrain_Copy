"""
Created on Mon Nov  4 14:50:00 2019

Author: Viktor Loreth

This file is used to run the utils.py file on all images in a given path.
It is used to generate the Saliency maps for the images in MNIST

"""
import sys
import torchvision
import os
import torch
from torchvision import transforms
from captum.attr import IntegratedGradients
# load model from ../Training/Train_Mnist.py
from Training.Train_MNIST_SimpleCNN_original_Dataset import Net
import numpy as np
import time

sys.path.insert(0, '../Training')

# torch seed
torch.manual_seed(0)

# Windows path:
pathtrain = 'C:\\Users\Vik\Documents\\4. Private\\01. University\\2023_Sem6\\Intepretable_AI\\data\\MNIST\\mod_imagestrain'
pathtest = 'C:\\Users\Vik\Documents\\4. Private\\01. University\\2023_Sem6\\Intepretable_AI\\data\\MNIST\\mod_imagestest'

for path in [pathtrain, pathtest]:
	
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print(device)
	torch.cuda.empty_cache()
	torch.cuda.synchronize()
	
	thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
	
	model = Net()
	model.load_state_dict(torch.load('../models/mnist/integrated_gradients/models/original_net.pth'))
	model.to(device)
	model.eval()
	
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
			img = img.to(device)
			# save the label in the logfile and end the line
			f.write(f'{i}, {label} \n')
			
			# save the 0 image in folder 0
			torchvision.utils.save_image(img, path + f'\\0\\{label}\\{i}.png')
			
			ig = IntegratedGradients(model)
			# plot the saliency map ig
			ig_attr = ig.attribute(img.unsqueeze(0), target=label)
			# flatten ig_attr to 1D
			ig_attr_flat = torch.abs(ig_attr.view(-1))
			
			# find topk indices of most important pixels of ig_attr_flat
			indices = torch.topk(ig_attr_flat, int(len(ig_attr_flat) * 0.91))[1]
			
			for ii in range(len(thresholds)):
				mask = torch.zeros(28, 28, dtype=torch.bool).to(device)
				indices_tmp = indices.clone()
				# copy the original image
				img_masked = img.clone()
				# split up indices to only take the top threshold % of pixels
				indices_tmp = indices_tmp[:int(len(ig_attr_flat) * thresholds[ii])]
				
				indices_tmp = torch.stack((indices_tmp // 28, indices_tmp % 28), dim=1)
				
				# set the mask to True for the top threshold % of pixels
				mask[indices_tmp[:, 0], indices_tmp[:, 1]] = True
				
				# set the pixels of the image to mean value of MNIST numbers
				img_masked[:, mask] = 0.1307
				
				# save the image
				torchvision.utils.save_image(img_masked, path + f'\\{thresholds[ii]}\\{label}\\{i}.png')
			# track progress
			if i % 1000 == 0:
				print(f'{i} images processed')
				# print timer progress
				print(f'{time.time() - start} seconds per image')
	
	print("success")
