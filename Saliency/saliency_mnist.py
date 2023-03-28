"""
Created on Mon Nov  4 14:50:00 2019

Author: Viktor Loreth

This file is used to run the saliency_transform.py file on all images in a given path.
It is used to generate the Saliency maps for the images in MNIST
The files are saved in a folder called ILSVRC30, ILSVRC50, ILSVRC70 respectively.

"""
import sys
import torchvision
import os
import torch
from torchvision import transforms
from captum.attr import IntegratedGradients

sys.path.insert(0, '../Training')
# Windows path:
path = 'C:\\Users\Vik\Documents\\4. Private\\01. University\\2023_Sem6\\Intepretable_AI\\data\\MNIST\\mod_images'

use_cuda = False # torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.cuda.empty_cache()
torch.cuda.synchronize()

thresholds = [0.3, 0.5, 0.7]

# load model from ../Training/Train_Mnist.py
from Training.Train_MNIST import Net

model = Net()
model.load_state_dict(torch.load('../models/mnist/original_net.pth'))
model.to(device)
model.eval()

# Define the transform to apply to the input images
transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.1307,), (0.3081,))])

# Load the training and testing data
trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                      download=True, transform=transform)

print("Data loaded")
# generate a folder for each threshold
if not os.path.exists(path + '\\0.3'):
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
		# get the image and the label
		img, label = trainset[i]
		# calculate the integrated gradients
		ig = IntegratedGradients(model)
		
		
print("success")
