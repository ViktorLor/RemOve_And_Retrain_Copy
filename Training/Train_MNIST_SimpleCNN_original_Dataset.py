"""
Trains MNIST on the official dataset. The model is saved in the models folder.
Does 5 evaluations.
Only 1 model is saved.
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
	# define how many runs
	runs = 5
	
	# Define the transform to apply to the input images
	transform = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize((0.1307,), (0.3081,))])
	
	# Load the training and testing data
	trainset = torchvision.datasets.MNIST(root='../data', train=True,
	                                      download=True, transform=transform)
	
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
	                                          shuffle=True)
	
	testset = torchvision.datasets.MNIST(root='../data', train=False,
	                                     download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=64,
	                                         shuffle=False)
	
	for run in range(runs):
		# initialize random seed:
		torch.manual_seed(run)
		# Initialize the CNN and define the loss function and optimizer
		net = Net()
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
		
		print("Start Training")
		# Train the CNN for 10 epochs
		for epoch in range(10):
			running_loss = 0.0
			for i, data in enumerate(trainloader, 0):
				inputs, labels = data
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
				outputs = net(images)
				_, predicted = torch.max(outputs, 1)
				c = (predicted == labels).squeeze()
				for i in range(4):
					label = labels[i]
					class_correct[label] += c[i].item()
					class_total[label] += 1
		
		# write accuracy to file
		with open('../models/mnist/original_accuracy.txt', 'a') as f:
			# mention which run it is:
			f.write('Run: %d \n' % run)
			# mention the seed
			f.write('Seed: %d \n' % torch.initial_seed())
			for i in range(10):
				f.write('Accuracy of %5s : %2d %%\n' % (
					str(i), 100 * class_correct[i] / class_total[i]))
		
		if run == 0:
			# Save the trained model
			PATH = f'../models/mnist/original_net.pth'
			torch.save(net.state_dict(), PATH)
