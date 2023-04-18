"""
Takes the MNIST trained models and generates a report for each model.
The report contains the following:
- A txt file with the accuracy of the model for each class, the standard deviation and the average accuracy
- A png file with the confusion matrix of the model
- A png file with the ROC curve of the model
"""
# import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch
import torchvision
from torchvision import transforms
import sys
# import train
sys.path.insert(0, '../Training')

from Training.Train_MNIST import Net
# define the function to plot the confusion matrix
def plot_confusion_matrix(cm, classes,threshold,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	:param cm:
	:param classes:
	:param normalize:
	:param title:
	:param cmap:
	:return:
	"""
	
	if normalize:
		# convert Tesnor to numpy array
		cm = cm.numpy()
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')
		
	
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
		         horizontalalignment="center",
		         color="white" if cm[i, j] > thresh else "black")
		
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	# save the figure
	plt.savefig(f'../models/mnist/confusion_matrix_{threshold}.png')
	plt.close()
	

path = '../models/mnist'

# load the models
thresholds = [0.1, 0.3, 0.5,0.7, 0.9]

transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.1307,), (0.3081,)),
	 transforms.Lambda(lambda x: x[0].unsqueeze(0))])

for threshold in thresholds:
	# load weights
	weights = torch.load(path + f'\\{threshold}_net.pth')
	
	# define the model
	model = Net()
	# load the weights
	model.load_state_dict(weights)
	
	
	# load testdata
	testset = torchvision.datasets.ImageFolder(root=f'../data/MNIST/mod_imagestest/{threshold}',
	                                           transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=64,
	                                         shuffle=False)
	
	# get the accuracy of the model
	# define the confusion matrix
	confusion_matrix = torch.zeros(10, 10, dtype=torch.int32)
	# define the accuracy of each class
	accuracy = torch.zeros(10, dtype=torch.float32)
	# define the number of images of each class
	n_images = torch.zeros(10, dtype=torch.int32)
	
	# iterate over the test data
	for images, labels in testloader:
		# get the predictions
		predictions = model(images)
		# get the class with the highest probability
		_, predicted = torch.max(predictions, 1)
		# update the confusion matrix
		for t, p in zip(labels.view(-1), predicted.view(-1)):
			confusion_matrix[t.long(), p.long()] += 1
		# update the accuracy in %
		for i in range(len(labels)):
			n_images[labels[i]] += 1
			if labels[i] == predicted[i]:
				accuracy[labels[i]] += 1

	
	# calculate the accuracy in %
	accuracy = accuracy / n_images * 100
	# calculate the average accuracy
	avg_accuracy = torch.mean(accuracy)
	# calculate the standard deviation
	std = torch.std(accuracy)
	
	# save the accuracy of each class, the standard deviation and the average accuracy to a txt file
	with open(path + f'\\{threshold}_accuracy.txt', 'w') as f:
		f.write(f'Accuracy of each class: {accuracy}\n')
		f.write(f'Standard deviation: {std}\n')
		f.write(f'Average accuracy: {avg_accuracy}\n')
		
	# plot the confusion matrix
	if True:
		plot_confusion_matrix(confusion_matrix, range(10), threshold, normalize=True)
	
