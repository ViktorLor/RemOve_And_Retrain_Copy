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
import torchvision.transforms as transforms

####### TESTING MASKING GENERATION: TORCH TENSOR#######

path_mask = r'C:\Users\Vik\Documents\4. Private\01. University\2023_Sem6\Intepretable_AI\data\food-101\indices_to_block\integrated_gradient\apple_pie\134.png'
img_path= r'C:\Users\Vik\Documents\4. Private\01. University\2023_Sem6\Intepretable_AI\data\food-101\images\apple_pie\134.jpg'



image = Image.open(img_path).convert('RGB')
mask = Image.open(path_mask).convert('RGB')


transformer = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.561, 0.440, 0.312], std=[0.252, 0.256, 0.259])
])

image = transformer(image)
#convert mask to tensor
mask = transforms.ToTensor()(mask)

mean_1, mean_2, mean_3 = 0.485, 0.456, 0.406

threshold = 0.5
# Apply mask to image . the division by 255 is necessary because the mask is saved as a png file and therefore has values between 0 and 255.
image[0, :, :] = torch.where(mask[0, :, :] < (threshold / 255), image[0, :, :], torch.tensor(mean_1))
image[1, :, :] = torch.where(mask[1, :, :] < (threshold / 255), image[1, :, :], torch.tensor(mean_2))
image[2, :, :] = torch.where(mask[2, :, :] < (threshold / 255), image[2, :, :], torch.tensor(mean_3))

# TO control if the mask is applied correctly
import matplotlib.pyplot as plt
plt.imshow(image.permute(1, 2, 0))
plt.show()


