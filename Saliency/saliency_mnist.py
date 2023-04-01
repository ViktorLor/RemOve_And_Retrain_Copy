"""
Created on Mon Nov  4 14:50:00 2019

Author: Viktor Loreth

This file is used to run the saliency_transform.py file on all images in a given path.
It is used to generate the Saliency maps for the images in MNIST

"""
import sys
import torchvision
import os
import torch
from torchvision import transforms
from captum.attr import IntegratedGradients
# load model from ../Training/Train_Mnist.py
from Training.Train_MNIST import Net

sys.path.insert(0, '../Training')

# do it once for path training and path testing


# Windows path:
pathtrain = 'C:\\Users\Vik\Documents\\4. Private\\01. University\\2023_Sem6\\Intepretable_AI\\data\\MNIST\\mod_imagestrain'
pathtest = 'C:\\Users\Vik\Documents\\4. Private\\01. University\\2023_Sem6\\Intepretable_AI\\data\\MNIST\\mod_imagestest'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.cuda.empty_cache()
torch.cuda.synchronize()

thresholds = [0.1, 0.5, 0.9]



model = Net()
model.load_state_dict(torch.load('../models/mnist/original_net.pth'))
model.to(device)
model.eval()


# Define the transform to apply to the input images
transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.1307,), (0.3081,))])

for path in [pathtrain, pathtest]:

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
	
	print("Folders created")
	# open a logfile to save the labels
	with open(path + '\\labels.txt', 'w') as f:
		# iterate over all images in the trainset
		for i in range(len(trainset)):
			
			# get the image an the label
			img, label = trainset[i]
			# send img to device
			img = img.to(device)
			# save the label in the logfile and end the line
			f.write(f'{i}, {label} \n')
			
			# save the 0 image in folder 0
			torchvision.utils.save_image(img, path + f'\\0\\{label}\\{i}.png')
			
			# calculate the integrated gradients
			ig = IntegratedGradients(model)
			# plot the saliency map ig
			ig_attr = ig.attribute(img.unsqueeze(0), target=label)
			# flatten ig_attr to 1D
			ig_attr_flat = ig_attr.view(-1)
			
			for ii in range(len(thresholds)):
				# copy the original image
				img_masked = img.clone()
				
				# find topk indices of most important pixels of ig_attr_flat
				indices = torch.topk(ig_attr_flat, int(len(ig_attr_flat) * thresholds[ii]))[1]
				# split up indices into 2D indices
				indices = torch.stack((indices // 28, indices % 28), dim=1)
				
				
				# set the pixels of the image to mean value of MNIST numbers
				for j in range(len(indices)):
					img_masked[0, indices[j][0], indices[j][1]] = 0.1307
				
				# save the image
				torchvision.utils.save_image(img_masked, path + f'\\{thresholds[ii]}\\{label}\\{i}.png')
			# track progress
			if i % 1000 == 0:
				print(f'{i} images processed')
			
	print("success")
