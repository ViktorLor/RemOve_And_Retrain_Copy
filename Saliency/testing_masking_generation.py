"""
RESULT: PT AND PIL IMAGE ARE THE SAME

NO ADDITIONAL TESTING REQUIRED. MASKING WORKS AS EXPECTED


"""

# import libraries
import os
import numpy as np
import PIL
from PIL import Image
import torch
import matplotlib.pyplot as plt

####### TESTING MASKING GENERATION: TORCH TENSOR#######

path = r'C:\Users\Vik\Documents\4. Private\01. University\2023_Sem6\Intepretable_AI\data\food-101\indices_to_block\random_baseline\apple_pie\1005649.pt'

# torch tensor laod from path
image_array = torch.load(path)

print(image_array.shape)
for i in [0, 1, 2]:
	# should be different for each channel
	print(torch.unique(image_array[i], return_counts=True))

# plt.imshow(image_array)
# plt.show()

####### TESTING MASKING GENERATION: PIL IMAGE #######
path = r'C:\Users\Vik\Documents\4. Private\01. University\2023_Sem6\Intepretable_AI\data\food-101\indices_to_block\random_baseline\apple_pie\1005649.png'

# PIL image load from path
image = Image.open(path)

# convert to rgb
image = image.convert('RGB')
# convert to numpy array
image_array = np.array(image)
# convert to 3x224x224
image_array = np.transpose(image_array, (2, 0, 1))
print(image_array.shape)
for i in [0,1,2]:
	print(np.unique(image_array[i,:,:], return_counts=True))