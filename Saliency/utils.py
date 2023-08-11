# coding=utf-8
# Copyright 2023 Viktor Loreth
"""
food_101_params = {
    'train_batch_size': 256,
    'num_train_images': 75750,
    'num_eval_images': 25250,
    'num_label_classes': 101,
    'num_train_steps': 20000,
    'base_learning_rate': 0.7,
    'weight_decay': 0.0001,
    'eval_batch_size': 256,
    'mean_rgb': [0.561, 0.440, 0.312],
    'stddev_rgb': [0.252, 0.256, 0.259]
}
"""

"""Saliency helper library to compute and pre-process saliency heatmaps.
The executable part of this file is used to test the functions. For the actual use of the functions,
please import the functions from this file."""

# Integrated Gradients, Gradient Saliency, Guide Backpropagation are used to generate saliency maps

from captum.attr import IntegratedGradients, GuidedBackprop
from matplotlib.colors import LinearSegmentedColormap
import torch
import numpy as np
from PIL import Image
import os
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
import time
import PIL


def generate_random_masks_3D(dataset_size, path_to_dataset, save_folder, images_to_copy='images', image_size=224,
                             saveaspng=False):
	"""
	Generate random indce masks which block out [0.1,0.3,0.5,0.7,0.9%] of the image.

	:param dataset_size: how many masks to create
	:param image_size: which shape the masks should have [224,224]
	:return:

	"""
	
	# Define the values and their respective probabilities
	values = [0, 1, 2, 3, 4, 5]
	percentages = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
	
	# Calculate the number of elements based on percentages
	total_elements = 3 * image_size * image_size  # You can adjust this to your desired array size
	element_counts = np.round(np.array(percentages) * total_elements).astype(int)
	element_counts[0] -= 1  # Make sure the sum of the element counts equals the total elements
	element_counts[-1] -= 1
	print(element_counts.sum())
	
	# find all folders in the dataset
	folders = [f for f in os.listdir(path_to_dataset + images_to_copy) if not f.startswith('.')]
	
	for folder in folders:
		start = time.time()
		# generate folder if it does not exist
		if not os.path.exists(path_to_dataset + save_folder + folder):
			os.makedirs(path_to_dataset + save_folder + folder)
		# find all images in the folder
		images = [f for f in os.listdir(path_to_dataset + images_to_copy + '/' + folder) if not f.startswith('.')]
		
		# Create the deterministic array
		random_array = np.concatenate(
			[np.full(count, value, dtype=np.uint8) for value, count in zip(values, element_counts)])
		
		for image in images:
			# Shuffle the array to ensure randomness
			np.random.shuffle(random_array)
			
			# conver np array to 3x224x224 array
			tmp_array = np.reshape(random_array, (3, image_size, image_size))
			# save array as tensor
			tmp_array = torch.from_numpy(tmp_array)
			
			# save tensor
			if saveaspng:
				tmp_array = tmp_array.numpy()
				tmp_array = np.transpose(tmp_array, (1, 2, 0))
				tmp_array = PIL.Image.fromarray(tmp_array)
				tmp_array.save(path_to_dataset + save_folder + folder + '/' + image[:-4] + '.png')
			else:
				torch.save(tmp_array, path_to_dataset + save_folder + folder + '/' + image[:-4] + '.pt')
		
		print("Folder " + folder + " done.")
		print("Time elapsed: " + str(time.time() - start) + " seconds.")
	
	print("Finished generating random masks.")
	
	return


def generate_random_masks_3D_memmap(dataset_size, path_to_dataset, save_file, image_size=224):
	"""
	Generate random indce masks which block out [0.1,0.3,0.5,0.7,0.9%] of the image. Saves in a memmap file.
	:param dataset_size:
	:param path_to_dataset:
	:param save_file:
	:param image_size:
	:return:

	NOT TESTED YET; NOT SURE IF IT WORKS; IS FASTER BUT COSTS MORE MEMORY
	"""
	
	masks_memmap = np.memmap(f"{path_to_dataset}/{save_file}.dat", dtype=np.uint8, mode='w+',
	                         shape=(dataset_size, image_size, image_size))
	
	# Define the values and their respective probabilities
	values = [0, 1, 2, 3, 4, 5]
	percentages = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
	
	# Calculate the number of elements based on percentages
	total_elements = 3 * image_size * image_size  # You can adjust this to your desired array size
	element_counts = np.round(np.array(percentages) * total_elements).astype(int)
	element_counts[0] -= 1  # Make sure the sum of the element counts equals the total elements
	element_counts[-1] -= 1
	print(element_counts.sum())
	
	# Create the deterministic array
	random_array = np.concatenate(
		[np.full(count, value, dtype=np.uint8) for value, count in zip(values, element_counts)])
	
	start = time.time()
	for x in range(dataset_size):
		
		np.random.shuffle(random_array)
		masks_memmap[x] = np.reshape(random_array, (3, image_size, image_size))
		
		if x == 1000:
			print("Time elapsed: " + str(time.time() - start) + " seconds.")
			print("Total time for 1001 sets: " + str((time.time() - start) * 101 / 60) + " minutes.")
		
		if x % 1000 == 0:
			print(f"Generated {x} masks")


def generate_saliency_masks_3D(model, method, path_to_dataset, image_size=224, test=True, saveaspng=False):
	"""
	Right now works only for food101 dataset
	
	:param model:
	:param method:
	:param path_to_dataset:
	:param image_size:
	:return:
	"""
	if test:
		print("TEST MODE IS ON!!! USING RANDOM MASKS! IF YOU HAVE A FUNCTIONAL MODEL, TURN TEST MODE OFF!")
	
	# Check if torch is available
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print(device)
	torch.cuda.empty_cache()
	torch.cuda.synchronize()
	
	thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
	
	# Load the model
	model.to(device)
	model.eval()
	
	# Define the transform to apply to the input images
	transformer = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.561, 0.440, 0.312], std=[0.252, 0.256, 0.259])
	])
	
	# load metadata to convert label to names
	with open(path_to_dataset + '\\meta\\classes.txt', 'r') as f:
		classes = f.readlines()
		classes = [c.replace('\n', '') for c in classes]
		# replace space with underscore
		classes = [c.replace(' ', '_') for c in classes]
	
	# create a folder for each class
	for c in classes:
		os.makedirs(path_to_dataset + f'indices_to_block\\{method}\\{c}', exist_ok=True)
	
	### load images
	# find all folders in the dataset
	folders = [f for f in os.listdir(path_to_dataset + 'images') if not f.startswith('.')]
	
	for z, folder in enumerate(folders):
		images = os.listdir(path_to_dataset + 'images/' + folder)
		label = classes.index(folder)
		for i, image in enumerate(images):
			start = time.time()
			try:
				
				img_data = Image.open(path_to_dataset + 'images\\' + folder + '\\' + image)
				img_data = transformer(img_data)
				generate_singular_saliency_3D_mask(img_data, label, model, method, path_to_dataset, folder, image,
				                                   image_size=224,
				                                   test=test, saveaspng=saveaspng)
			
			except Exception as e:
				print(e)
				print("Error with image: ", image)
				# write to logfile
				with open("logfile.txt", "a") as logfile:
					logfile.write("Error with image: " + image)
			
			if i % 100 == 0:
				print(i, z, "image done", len(folders))
			
			if i == 1000:
				print("Time elapsed: " + str(time.time() - start) + " seconds.")
				print("Total time for 1001 sets: " + str((time.time() - start) * 101 / 60) + " minutes.")
				print("Do you want to continue? (y/n)")
				choice = input().lower()
				if choice == 'n':
					exit(1)


def generate_singular_saliency_3D_mask(img, label, model, method, path_to_dataset, folder, img_name, image_size=224,
                                       test=True, saveaspng=False):
	if test == False:
		
		ig = IntegratedGradients(model)
		# get the ig_attr for the image
		ig_attr = ig.attribute(img.unsqueeze(0), target=label)
		# flatten ig_attr to 1D array = 244*224*3
		ig_attr_flat = torch.abs(ig_attr).flatten()
		mask = torch.zeros_like(ig_attr, dtype=torch.uint8)
	else:
		
		ig_attr_flat = torch.abs(torch.rand((image_size * image_size * 3)))
		mask = torch.zeros((3, image_size, image_size), dtype=torch.uint8)
	
	# find topk indices of most important pixels of ig_attr_flat
	indices = torch.topk(ig_attr_flat, int(len(ig_attr_flat)))[1]
	
	new_indices = np.unravel_index(indices, (3, image_size, image_size))
	# convert tuple to numpy array
	new_indices = np.array(new_indices)
	
	for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
		# get array of indices to fill
		indices_to_fill = new_indices[:, :int(224 * 224 * 3 * threshold)]
		
		mask[indices_to_fill[0], indices_to_fill[1], indices_to_fill[2]] += 1
	
	# save mask as torch tensor compressed
	if saveaspng:
		tmp_array = mask.numpy()
		tmp_array = np.transpose(tmp_array, (1, 2, 0))
		tmp_array = PIL.Image.fromarray(tmp_array)
		tmp_array.save(path_to_dataset + f'indices_to_block\\{method}\\{folder}\\{img_name[:-4]}.png')
	else:
		torch.save(mask, path_to_dataset + f'indices_to_block\\{method}\\{folder}\\{img_name[:-4]}.pt')


def get_saliency_image(model, y, image, saliency_method):
	"""generates saliency image.
		Args:
		  model: model to compute saliency maps.
		  y: the pre-softmax activation we want to assess attribution with respect to.
		  image: float32 image tensor with size [1, None, None].
		  saliency_method: string indicating saliency map type to generate.
		Returns:
		  a saliency map and a smoothed saliency map.
		Raises:
		  ValueError: if the saliency_method string does not match any included method
		"""
	
	if saliency_method == "integrated_gradients":
		integrated_placeholder = IntegratedGradients(model)
		return integrated_placeholder.attribute(image, target=y)
	
	elif saliency_method == "guided_backprop":
		gb_placeholder = GuidedBackprop(model)
		return gb_placeholder.attribute(image, target=y)
	
	else:
		raise ValueError("No saliency method method matched. Verification of"
		                 "input needed")
