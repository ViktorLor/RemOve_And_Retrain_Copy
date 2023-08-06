"""
New utils file as the old one is deprecated.


"""
import numpy as np
import os
from PIL import Image
import time


def generate_random_masks_3D(dataset_size, path_to_dataset, save_folder, images_to_copy='images', image_size=224):
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
			tmp_array = np.reshape(random_array, (image_size, image_size, 3))
			
			# save the array as png file
			my_array = Image.fromarray(tmp_array)
			
			my_array.save(path_to_dataset + save_folder + folder + '/' + image[:-4] + '.png')
		
		print("Folder " + folder + " done.")
		print("Time elapsed: " + str(time.time() - start) + " seconds.")
		if folder == 'apple_pie':
			print("Time for 1001 sets: " + str((time.time() - start) * 101 / 60) + " minutes.")
			
			print("You want to continue? [y/n]")
			choice = input().lower()
			if choice == 'n':
				exit(1)
	
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
