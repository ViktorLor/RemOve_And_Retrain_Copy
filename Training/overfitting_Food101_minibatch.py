"""
Training the ResNet-50 model on the Food-101 dataset.

Parameters used: https://github.com/google-research/google-research/blob/master/interpretability_benchmark/train_resnet.py#L113

30h Training by now.
Probably possible to crease batch size to increase speed; -> But then paper is not completely reproducible
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torchvision import models
import os
import time
import utils
from torch.utils.tensorboard import SummaryWriter

# define seed
torch.manual_seed(0)

# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# Define transforms for training and validation data
transformer = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.561, 0.440, 0.312], std=[0.252, 0.256, 0.259])
])

# Load the Food 101 dataset train and test with the original split of the dataset
train_dataset = torchvision.datasets.Food101(root='../Data', transform=transformer, download=True, split="train")

print("Loaded Dataset")
print("Train Dataset: ", len(train_dataset))
print("Size should be: ", 75750)

batch_size = 64  # 80 seems to fit in the memory of the GPU
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                          pin_memory=True,
                                          prefetch_factor=4)

# Load a randomly initialized ResNet50 model with mü = 0 and σ = 0.01
model = models.resnet50()
# Replace the last layer with a new fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 101)

model.apply(lambda module: utils.initialize_weights(module, mean=0, std=0.01))

print("Model Initialized")

# Send the model to the device
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()

initial_learning_rate = 0.07
# training steps 20000
learning_rate = initial_learning_rate * float(batch_size) / 256  # adjusting according to the paper
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)

print("started training model")
start = time.time()
# Train the model
num_epochs = 90

writer = SummaryWriter(log_dir='../models/food101/runs_original/')

accuracy = 0

for i, data in enumerate(data_loader, 0):
	for epoch in range(num_epochs):
		
		# print epoch
		print("Epoch: ", epoch + 1)
		
		running_loss = 0.0
		
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
		accuracy += (predicted == labels).sum().item() / batch_size
		
		loss.backward()
		optimizer.step()
		scheduler.step()
		
		# Print statistics
		running_loss += loss.item()
		
		if (accuracy == 1):
			print("Accuracy: ", accuracy)
			print("Epoch: ", epoch + 1)
			exit(1)
		
		# print running loss
		print('[%d, %5d] loss: %.3f' %
		      (epoch + 1, i + 1, running_loss))
		running_loss = 0.0
		# print accuracy
		print("Accuracy: ", accuracy)
		accuracy = 0.0
	
	break
