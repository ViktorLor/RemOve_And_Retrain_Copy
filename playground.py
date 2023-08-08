
# import libraries
import os
import numpy as np
import PIL
from PIL import Image
import torch
import matplotlib.pyplot as plt

path = r'C:\Users\Vik\Documents\4. Private\01. University\2023_Sem6\Intepretable_AI\data\food-101\indices_to_block\integrated_gradient\apple_pie\101251.pt'

# torch tensor laod from path
image_array = torch.load(path)

print(image_array.shape)

for i in [0,1,2]:
	print(torch.unique(image_array[i], return_counts=True))
	



#plt.imshow(image_array)
#plt.show()

exit(1)