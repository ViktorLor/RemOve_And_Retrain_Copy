"""
Training the ResNet-50 model on the Food-101 dataset.

Parameters used: https://github.com/google-research/google-research/blob/master/interpretability_benchmark/train_resnet.py#L113

"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torchvision import models
import os
import time

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
test_dataset = torchvision.datasets.Food101(root='../Data', transform=transformer, download=True, split="test")
print("Loaded Dataset")
print("Train Dataset: ", len(train_dataset))
print("Test Dataset: ", len(test_dataset))

# Create dataloaders for training and test data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load a randomly initialized ResNet50 model
model = models.resnet50()
# weights are initialied by drawing weights from a zero mean gaussian with mü=0 and sigma=0.01
model.parameters().data.normal_(0, 0.01)

print("Model Initialized")
# Replace the last layer with a new fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 101)

# Send the model to the device
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
# Parameters from: arXiv:1706.02677v2 [cs.CV] 30 Apr 2018; Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour
# settings: Nesterov momentum m=0.9, weight_decay=0.0001, no weight_decay on lambda and beta
# BN batch_size n=32, 90 epochs, lr=0.1* batch_size/256 [reference_learning_rate], reduce lr by 1/10 at epoch 30, 60, 80
# learnable scaling parameter gamma =1, except for the last BN in each block, where it is 0.

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)

print("started training model")
start = time.time()
# Train the model
num_epochs = 90

for epoch in range(num_epochs):
	running_loss = 0.0
	# print epoch
	print("Epoch: ", epoch + 1)
	
	for i, data in enumerate(train_loader, 0):
		try:
			# Get the inputs and labels
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)
			
			# Zero the parameter gradients
			optimizer.zero_grad()
			
			# Forward + backward + optimize
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			scheduler.step()
			
			# Print statistics
			running_loss += loss.item()
			if i % 10 == 0 and i != 0:  # print every 10 mini-batches
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
				running_loss = 0.0
			
			if i == 100:
				# print how long the training will take for 1 epoch
				end = time.time()
				print("Estimated training time for 1 epoch: ", len(train_loader) / 100 * (end - start) / 60, " minutes")
				print("1 epoch will be done at: ", time.ctime(end + (end - start)))
		
		except:
			# write in log file that error occured
			with open('../models/food101/original_training_log.txt', 'a') as f:
				f.write("Error in epoch: " + str(epoch) + " batch: " + str(i) + "\n")
				# save model, optimizer and scheduler
				torch.save(model.state_dict(), '../models/food101/ResNet50_Food101_original_aborted.pth')
				torch.save(optimizer.state_dict(), '../models/food101/ResNet50_Food101_original_aborted_optimizer.pth')
				torch.save(scheduler.state_dict(), '../models/food101/ResNet50_Food101_original_aborted_scheduler.pth')
			# continue training
			continue
	
	if epoch == 0:
		end = time.time()
		print("Estimated training time: ", (end - start) * num_epochs / 60, " minutes")
		print("Training will be done at: ", time.ctime(end + (end - start) * num_epochs))
		# ask if you want to continue training
		answer = input("Do you want to continue training? (y/n)")
		if answer == "n":
			print("Training stopped")
			exit(1)

print('Finished Training')

# check if folder ../models/food101 exists
if not os.path.exists('../models/food101'):
	os.makedirs('../models/food101')

# save the model
torch.save(model.state_dict(), '../models/food101/ResNet50_Food101_original.pth')

# test the model and the accuracy
correct = [0 for i in range(101)]
total = [0 for i in range(101)]

with torch.no_grad():
	for data in test_loader:
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
	with open('../models/food101/original_accuracy.txt', 'w') as f:
		for i in range(101):
			if total[i] != 0:
				f.write('Accuracy of %5s : %2d %%\n' % (
					train_dataset.dataset.classes[i], 100 * correct[i] / total[i]))
			else:
				f.write('Accuracy of %5s : %2d %%\n' % (
					train_dataset.dataset.classes[i], 0))
