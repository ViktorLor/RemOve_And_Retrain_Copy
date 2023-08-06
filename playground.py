
from PIL import Image
import numpy as np
# Load image from path
path = r'C:\Users\Vik\Documents\4. Private\01. University\2023_Sem6\Intepretable_AI\data\food-101\indices_to_block\random_baseline\apple_pie\134.png'

image = Image.open(path)

# Convert image to numpy array
image = np.array(image)

# load unique values
unique_values = np.unique(image, return_counts=True)

# print unique values
print(unique_values)
