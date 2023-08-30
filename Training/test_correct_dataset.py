import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torchvision import models
import os
import time
import utils
import sys

# define seed
torch.manual_seed(0)

# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
# Define transforms for training and validation data
transformer = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.561, 0.440, 0.312], std=[0.252, 0.256, 0.259])
])

# Load first parameter "generate_random_masks"
# either "random_baseline" or "integrated_gradient"
folder = sys.argv[1]
print(folder)

if folder != "random_baseline" and folder != "integrated_gradient" and folder != "guided_backprop":
	print("Wrong input")
	exit()

food101path = '../Data/food-101'
# threshold: threshold: >4.5: 10%, 3.5: 30%, 2.5: 50%, 1.5: 70%, 0.5: 90%,masked
threshold_to_string = {4.5: "10", 3.5: "30", 2.5: "50", 1.5: "70", 0.5: "90"}

for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:  # add 3.5 and 4.5 if successful
	train_dataset = utils.Food101MaskDataset(data_folder_images=food101path + '/images/',
	                                         data_folder_mask=food101path + '/indices_to_block/' + folder + '/',
	                                         meta_file=food101path + '/meta/train.txt',
	                                         threshold=threshold, transform=transformer)
	
	# mkdir test
	if not os.path.exists('../models/food101/' + 'test/' + threshold_to_string[threshold]):
		os.makedirs('../models/food101/' + 'test/' + threshold_to_string[threshold])
	
	#instantiate loader
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
	# save each singular image in folder
	for i, data in enumerate(train_loader, 0):
		images, labels = data
		for j in range(images.shape[0]):
			# convert tensor to image
			img = torchvision.transforms.ToPILImage()(images[j])
			# save image
			img.save('../models/food101/' + 'test/' + threshold_to_string[threshold] + '/' + str(i) + '_' + str(j) + '.jpg')
		
		break
		
