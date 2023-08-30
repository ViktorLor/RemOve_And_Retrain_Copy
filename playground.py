
# import libraries
import os
import numpy as np
import PIL
from PIL import Image
import torch
import matplotlib.pyplot as plt

# calaculate std and mean

path =r'C:\Users\Vik\Documents\4. Private\01. University\2023_Sem6\Intepretable_AI\models\food101\runs_original\original_ResNet50_lr_0_7_'

list=  []
for i in range(5):
	path2 = path + str(i) + '.csv'
	#load file
	data = np.genfromtxt(path2, delimiter=',')
	# only use last row
	data = data[-1]
	# only use last value of last row
	data = data[-1]
	list.append(data)

# calculate mean and std
mean = np.mean(list)
std = np.std(list)

print(mean)
print(std)