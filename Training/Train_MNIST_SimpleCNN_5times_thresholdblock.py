"""
Trains MNIST on a modified dataset with missing pixels.
The outputs are 5*5 models, one for each threshold and 5 for each run to get a better estimate of the accuracy.

"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim


# Define the CNN architecture
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 4 * 4, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
	
	def forward(self, x):
		x = self.pool(torch.relu(self.conv1(x)))
		x = self.pool(torch.relu(self.conv2(x)))
		x = x.view(-1, 16 * 4 * 4)
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)
		return x


if __name__ == "__main__":
	config = "randombaseline"
	
	if config == "ig":
		path = f"../data/MNIST/mod_images"
	elif config == "randombaseline":
		path = f"../data/MNIST/randombaseline_images"
	else:
		print("Invalid config")
		exit(1)
	runs = 5
	#device
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	# define seed
	torch.manual_seed(0)
	
	# define threshold
	thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
	
	# Define the transform to apply to the input images, transform the tensor to a 2d images
	# normalize the images and add a channel dimension which is needed for the CNN
	transform = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize((0.1307,), (0.3081,)),
		 transforms.Lambda(lambda x: x[0].unsqueeze(0))])
	
	# if models folder does not exist, create it
	import os
	
	if not os.path.exists('../models/mnist'):
		os.makedirs('../models/mnist')
	else:
		
		print("Models folder already exists")
		if input("Do you want to continue? (y/n)") == "n":
			exit(1)
	
	for threshold in thresholds:
		for run in range(runs):
			#set the seed
			torch.manual_seed(run)
			
			print(f"Run: {run}")
			print(f"Training for threshold {threshold}")
			# Load the training and testing data
			# the format in the txt file is: index, label
			# the images are in the folder: ../data/MNIST/mod_imagestrain/threshold
			# in each respective folder [0,1,2,3,4,5,6,7,8,9] the images are named: index.png

			trainset = torchvision.datasets.ImageFolder(root=path+f'train/{threshold}',
			                                            transform=transform)
			
			trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
			                                          shuffle=True)
			
			testset = torchvision.datasets.ImageFolder(root=path+f'test/{threshold}',
			                                           transform=transform)
			testloader = torch.utils.data.DataLoader(testset, batch_size=64,
			                                         shuffle=False)
			
			# Initialize the CNN and define the loss function and optimizer
			net = Net()
			criterion = nn.CrossEntropyLoss()
			optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
			#send to device
			net.to(device)
			
			print("Start Training")
			# Train the CNN for 10 epochs
			for epoch in range(10):
				running_loss = 0.0
				for i, data in enumerate(trainloader, 0):
					# get the inputs
					inputs, labels = data
					inputs = inputs.to(device)
					labels = labels.to(device)
					# zero the parameter gradients
					optimizer.zero_grad()
					outputs = net(inputs)
					loss = criterion(outputs, labels)
					loss.backward()
					optimizer.step()
					running_loss += loss.item()
					if i % 100 == 99:
						print('[%d, %5d] loss: %.3f' %
						      (epoch + 1, i + 1, running_loss / 100))
						running_loss = 0.0
			
			print('Finished Training')
			
			# Test the CNN on the test dataset, and print the accuracy for each class
			class_correct = list(0. for i in range(10))
			class_total = list(0. for i in range(10))
			with torch.no_grad():
				for data in testloader:
					images, labels = data
					images = images.to(device)
					labels = labels.to(device)
					outputs = net(images)
					_, predicted = torch.max(outputs, 1)
					c = (predicted == labels).squeeze()
					for i in range(4):
						label = labels[i]
						class_correct[label] += c[i].item()
						class_total[label] += 1
			
			# write accuracy to file
			if config == "ig":
				modelpath = f"../models/mnist/integrated_gradients/"
			elif config == "randombaseline":
				modelpath = f"../models/mnist/random_baseline/"
				
			with open(modelpath + f'{threshold}_accuracy.txt', 'a') as f:
				#mention the run and the seed
				f.write(f"Run: {run}, Seed: {run}\n")
				
				for i in range(10):
					f.write('Accuracy of %5s : %2d %%\n' % (
						str(i), 100 * class_correct[i] / class_total[i]))
				
			# Save the trained model
			PATH = modelpath +f'/models/{threshold}_{run}_net.pth'
			torch.save(net.state_dict(), PATH)
