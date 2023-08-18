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
import csv
import os
import time
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


# Mask Dataset which loads data from Food101 and additionally loads the masks and applies them to the images
class Food101MaskDataset(torch.utils.data.Dataset):
	def __init__(self, data_folder_images, data_folder_mask, meta_file, threshold, transform=None):
		# threshold: >4.5: 10%, 3.5: 30%, 2.5: 50%, 1.5: 70%, 0.5: 90%,masked
		try:
			self.threshold = threshold
			self.data_folder_images = data_folder_images
			self.data_folder_mask = data_folder_mask
			self.classes = os.listdir(data_folder_images)  # Get class names from subdirectories
			self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
			self.transform = transform
			
			# Create a list of (image_path,mask_path, label) pairs
			self.samples = []
			
			# Load meta file txt
			with open(meta_file, 'r') as f:
				self.meta = f.readlines()
			
			for line in self.meta:
				line = line.strip()
				img_path = os.path.join(data_folder_images, line + '.jpg')
				mask_path = os.path.join(data_folder_mask, line + '.png')
				idx = self.class_to_idx[line.split('/')[0]]
				
				self.samples.append((img_path, mask_path, idx))
		
		except Exception as e:
			print(self.samples)
			print("Error loading dataset: ", e)
			print("Please abort the program")
			quit(1)
	
	def __len__(self):
		return len(self.samples)
	
	def __getitem__(self, idx):
		# takes 8 ms per image, which is 11 min per epoch ->17.5 hours for 90 epochs, applying the mask takes ~1ms and is therefore negligible
		img_path, mask_path, label = self.samples[idx]
		
		image = Image.open(img_path).convert('RGB')
		mask = Image.open(mask_path).convert('RGB')
		
		mask = transforms.ToTensor()(mask)
		image = self.transform(image)
		
		mean_1, mean_2, mean_3 = 0.485, 0.456, 0.406
		
		# Apply mask to image . the division by 255 is necessary because the mask is saved as a png file and therefore has values between 0 and 255.
		image[0, :, :] = torch.where(mask[0, :, :] > (self.threshold / 255), image[0, :, :], torch.tensor(mean_1))
		image[1, :, :] = torch.where(mask[1, :, :] > (self.threshold / 255), image[1, :, :], torch.tensor(mean_2))
		image[2, :, :] = torch.where(mask[2, :, :] > (self.threshold / 255), image[2, :, :], torch.tensor(mean_3))
		
		# TO control if the mask is applied correctly
		# import matplotlib.pyplot as plt
		# plt.imshow(image.permute(1, 2, 0))
		# plt.show()
		
		return image, label


def training_food101(train_dataset, test_dataset, save_file, device, shuffle=True, seed=0):
	# check if folder ../models/food101 exists
	if not os.path.exists('../models/food101'):
		os.makedirs('../models/food101')
	
	# Set seed for reproducibility
	torch.manual_seed(seed)
	# Create dataloaders for training data
	
	batch_size = 64  # 80 seems to fit in the memory of the GPU
	traindata_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
	                                               num_workers=4,
	                                               pin_memory=True,
	                                               prefetch_factor=4)
	
	testdata_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
	                                              pin_memory=True,
	                                              prefetch_factor=4)
	
	# Load a randomly initialized ResNet50 model with mü = 0 and σ = 0.01
	model = models.resnet50()
	# Replace the last layer with a new fully connected layer
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, 101)
	# set model to training mode
	model.train()
	model.apply(lambda module: initialize_weights(module, mean=0, std=0.01))
	
	print("Model Initialized")
	
	# Send the model to the device
	model.to(device)
	
	# Define the loss function and optimizer
	criterion = nn.CrossEntropyLoss()
	
	initial_learning_rate = 0.7
	# training steps 20000
	learning_rate = initial_learning_rate * float(batch_size) / 256  # adjusting according to the paper
	optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)
	
	print("started training model")
	start = time.time()
	# Train the model
	num_epochs = 90
	
	writer = SummaryWriter(log_dir='../models/food101/runs_original/')
	
	accuracies_train = []
	running_losses = []
	
	for epoch in range(num_epochs):
		running_losses.append([])
		# print epoch
		print("Epoch: ", epoch + 1)
		accuracies_train.append([])
		running_loss = 0.0
		for i, data in enumerate(traindata_loader, 0):
			
			# Get the inputs and labels
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)
			
			# Zero the parameter gradients
			optimizer.zero_grad()
			
			# Forward + backward + optimize
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			# log accuracy
			_, predicted = torch.max(outputs.data, 1)
			accuracies_train[epoch].append((predicted == labels).sum().item() / batch_size)
			
			loss.backward()
			optimizer.step()
			
			# Print statistics
			running_loss += loss.item()
			
			if i % 100 == 0 and i != 0:  # print every 20 mini-batches
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / batch_size / 100))
				print('[%d, %5d] accuracy: %.3f' % (
					epoch + 1, i + 1, sum(accuracies_train[epoch]) / len(accuracies_train[epoch])))
				running_losses[epoch].append(running_loss / batch_size / 100)
				running_loss = 0.0
				# print accuracy and loss to tensorboard
				writer.add_scalar(f'Loss/train_p_batch', running_losses[epoch][-1], epoch * len(traindata_loader) + i)
				writer.add_scalar('Accuracy/train_p_batch',
				                  sum(accuracies_train[epoch]) / len(accuracies_train[epoch]),
				                  epoch * len(traindata_loader) + i)
				
				if i == 100 and epoch == 0:
					# print how long the training will take for 1 epoch
					end = time.time()
					print("Estimated training time for 1 epoch: ", (len(traindata_loader) / 100) * (end - start) / 60,
					      " minutes")
					print("Estimate training for 90 epochs: ", (len(traindata_loader) / 100) * (end - start) / 60 * 90,
					      " minutes")
					print("1 epoch will be done at: ", time.ctime(end + (end - start)))
		
		# update learning rate
		scheduler.step()
		
		# print accuracy and loss to tensorboard, every epoch
		writer.add_scalar(f'Loss/train_p_epoch', sum(running_losses[epoch]) / len(running_losses[epoch]), epoch)
		writer.add_scalar('Accuracy/train_p_epoch', sum(accuracies_train[epoch]) / len(accuracies_train[epoch]), epoch)
		
		total_correct = 0
		total_loss = 0
		
		model.eval()
		with torch.no_grad():
			for i, data in enumerate(testdata_loader, 0):
				images, labels = data
				images, labels = images.to(device), labels.to(device)
				outputs = model(images)
				loss = criterion(outputs, labels)
				total_loss += loss.item()
				
				_, predicted = torch.max(outputs.data, 1)
				total_correct += (predicted == labels).sum().item()
			
			writer.add_scalar('Accuracy/test_p_epoch', total_correct / len(test_dataset),
			                  epoch)
			writer.add_scalar('Loss/test_p_epoch', total_loss / len(test_dataset),
			                  epoch)
		model.train()
	
	# save tensorboard file
	writer.close()
	
	print('Finished Training')
	# save the model
	
	torch.save(model.state_dict(), '../models/food101/' + save_file + '.pth')
	print("Model saved")
	return


def test_food101(dataset, model, device, result_file, shuffle=True, seed=0):
	# Create dataloaders for training data
	data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=shuffle)
	
	# test the model and the accuracy
	correct = [0 for i in range(101)]
	total = [0 for i in range(101)]
	
	with torch.no_grad():
		for data in data_loader:
			images, labels = data
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			# save correct and total for each class
			for i in range(len(labels)):
				label = labels[i]
				correct[label] += (predicted[i] == labels[i]).item()
				total[label] += 1
		# calculate and print accuracy for each class
		with open('../models/food101/' + result_file, 'w') as f:
			for i in range(101):
				if total[i] != 0:
					f.write('Accuracy of %5s : %2d %%\n' % (
						dataset.classes[i], 100 * correct[i] / total[i]))
				else:
					f.write('Accuracy of %5s : %2d %%\n' % (
						dataset.classes[i], 0))


def initialize_weights(module, mean=0, std=0.01):
	if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
		torch.nn.init.normal_(module.weight, mean, std)
		if module.bias is not None:
			torch.nn.init.zeros_(module.bias)
