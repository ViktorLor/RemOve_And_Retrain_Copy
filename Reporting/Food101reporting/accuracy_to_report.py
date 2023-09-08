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

#
filepath = r'C:\Users\Vik\Documents\4. Private\01. University\2023_Sem6\Intepretable_AI\models\food101\runs_original'

####### original_runs #######
list = []
for i in range(3):
	with open(filepath + f'\\original_ResNet50_lr_0_7_{i}.csv', 'r') as file:
		# the accuracy is written in the last line
		accuracy = file.readlines()[-1].split()[-1]
		# accuracy is in the last column, seperator = ,
		accuracy = float(accuracy.split(',')[-1])
		list.append(accuracy)

# calculate mean and std
mean = np.mean(list)
std = np.std(list)  #
print(f'Accuracy of ResNet50: {mean:.2%} +- {std:.2%}')

#############################
####### random_baseline #####
#############################
filepath = r'C:\Users\Vik\Documents\4. Private\01. University\2023_Sem6\Intepretable_AI\models\food101\random_baseline'


for threshold in [10, 30, 50, 70, 90]:
	list = []
	for i in range(3):
		with open(filepath + f'\\{threshold}\\original_ResNet50_lr_0_7_{i}.csv', 'r') as file:
			# the accuracy is written in the last line
			accuracy = file.readlines()[-1].split()[-1]
			# accuracy is in the last column, seperator = ,
			accuracy = float(accuracy.split(',')[-1])
			list.append(accuracy)
	
	# calculate mean and std
	mean = np.mean(list)
	std = np.std(list)
	print("Random Baseline")
	print("Threshold: ", threshold)
	print(f'Accuracy of ResNet50: {mean:.2%} +- {std:.2%}')
#############################
####### integrated_gradients #
#############################

filepath = r'C:\Users\Vik\Documents\4. Private\01. University\2023_Sem6\Intepretable_AI\models\food101\random_baseline'

for threshold in [10, 30, 50, 70, 90]:
	list = []
	for i in range(3):
		with open(filepath + f'\\{threshold}\\original_ResNet50_lr_0_7_{i}.csv', 'r') as file:
			# the accuracy is written in the last line
			accuracy = file.readlines()[-1].split()[-1]
			# accuracy is in the last column, seperator = ,
			accuracy = float(accuracy.split(',')[-1])
			list.append(accuracy)
	
	# calculate mean and std
	mean = np.mean(list)
	std = np.std(list)
	print("Integrated Gradients")
	print("Threshold: ", threshold)
	print(f'Accuracy of ResNet50: {mean:.2%} +- {std:.2%}')