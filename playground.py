
# import libraries
import os
import numpy as np
import PIL
from PIL import Image
import torch
import matplotlib.pyplot as plt

path = '/home/viktorl/Intepretable_AI_PR_Loreth/Data/food-101/indices_to_block/integrated_gradient/tiramisu/1412012.png'

#load image
img = Image.open(path)
#display image on console
plt.imshow(img)
plt.show()

