"""
Author: Viktor Loreth
Date: 18.04.2023

This file is used to create a more extensive report out of the accuracy files.
Given the accuracy files of the models, this file creates a report with the following information:
- The average accuracy of the model
- The standard deviation of the model

- The average accuracy of all models combined
- The standard deviation of all models combined
- The total accuracy of all models combined
- The total standard deviation of all models combined
"""

import numpy as np

filepath = f'/models/mnist'
# choose: random_baseline, integrated_gradients
config = 'random_baseline'
filepath = filepath + f'/{config}/'

for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
	# load the .txt file
	# Initialize empty arrays to store the accuracy of each run for each class
	accuracy_per_run = []
	accuracy_per_class = [[] for _ in range(10)]
	
	# Parse the txt file and extract the accuracy values
	with open(filepath + f'{threshold}_accuracy.txt', 'r') as file:
		for line in file:
			if line.startswith('Accuracy'):
				# Extract the class index and accuracy value from the line
				class_index, accuracy = line.split(':')
				class_index = int(class_index.split()[-1])
				accuracy = float(accuracy.strip()[:-1]) / 100.0
				
				# Store the accuracy values
				accuracy_per_run.append(accuracy)
				accuracy_per_class[class_index].append(accuracy)
	
	# Calculate the mean and standard deviation of the accuracy per class
	mean_per_class = [np.mean(accuracy_per_class[i]) for i in range(10)]
	std_per_class = [np.std(accuracy_per_class[i]) for i in range(10)]
	# Calculate the mean and standard deviation of the accuracy for all runs combined
	mean_all_runs = np.mean(accuracy_per_run)
	std_run_accuracy = np.std([sum(accuracy_per_run[i * 10:i * 10 + 10]) / 10 for i in range(5)])
	
	with open(filepath + f'reports/{threshold}_report.txt', 'w') as file:
		# print mean accuracy per run
		for i in range(5):
			file.write(f'Accuracy of run {i}: {sum(accuracy_per_run[i * 10:i * 10 + 10])/10:.2%}\n')
		# calculate the std deviation of the run accuracy
		
		file.write(f'Standard deviation of run accuracy: {std_run_accuracy:.2%}\n')
		file.write(f'Mean accuracy for all runs combined: {mean_all_runs:.2%}\n\n')
		
		file.write('Mean accuracy per class:\n')
		for i in range(10):
			file.write(f'Class {i}: {mean_per_class[i]:.2%}\n')
		
		file.write('\nStandard deviation per class:\n')
		for i in range(10):
			file.write(f'Class {i}: {std_per_class[i]:.2%}\n')
		
		
		
