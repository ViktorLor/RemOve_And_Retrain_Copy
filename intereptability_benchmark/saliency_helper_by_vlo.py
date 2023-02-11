# coding=utf-8
# Copyright 2023 Viktor Loreth

"""Saliency helper library to compute and pre-process saliency heatmaps."""

# Integrated Gradients, Gradient Saliency, Guide Backpropagation are used to generate saliency maps

from captum.attr import IntegratedGradients, GuidedBackprop
from matplotlib.colors import LinearSegmentedColormap


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


def generate_masks( mask):
	"""
	:param mask: saliency mask
	:param threshold: how much of the image should be blacked out (0.5 = 50%)
	:return: a image with the blocked out saliency mask
	"""
	
	# convert the mask to a 1 channel image
	mask = mask[0].sum(dim=0).detach().numpy()

	# create mask 1 with 70% of the pixels blacked out
	# create mask 2 with 50% of the pixels blacked out
	# create mask 3 with 30% of the pixels blacked out

	mask1 = mask.copy()
	mask1 = mask1 > 0.001
	mask2 = mask.copy()
	mask2 = mask2 > 0.01
	mask3 = mask.copy()
	mask3 = mask3 > 0.05

	print(mask1.sum())
	print(mask2.sum())
	print(mask3.sum())

	return mask1, mask2, mask3


def apply_mask_to_image(image, mask):
	# convert True values to ImgNet mean
	# image = 1 *3*244*244
	# mask = 244*244
	# if mask is true, replace the pixel with the mean
	# if mask is false, keep the pixel
	# return image

	# apply mask to image
	image = image[0]
	image = image.permute(1, 2, 0)

	image[mask] = torch.tensor([0.485, 0.456, 0.406])

	image = image.permute(2, 0, 1)
	return image


if __name__ == '__main__':
	print("Running saliency Helper by vlo to test the functions")
	
	# test the functions
	import torch
	import torchvision
	import torchvision.transforms as transforms
	import torch.nn.functional as F
	import json
	
	model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
	model.eval()
	
	mean = [0.485, 0.456, 0.406]
	
	# Transformer always stays the same
	transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	
	# load imagenet_labels
	with open(
			'C:\\Users\\Vik\\Documents\\4. Private\\01. University\\2022_Sem5\\Intepretable_AI\\datasets\\imagenet_class_index.json') as f:
		imagenet_labels = json.load(f)
	imagenet_labels = {int(k): v for k, v in imagenet_labels.items()}
	
	from PIL import Image

	import time

	# time
	start = time.time()
	img_path = r'C:\Users\Vik\Documents\4. Private\01. University\2022_Sem5\Intepretable_AI\datasets\imagenet1000samples\n01491361_tiger_shark.JPEG'
	img = Image.open(img_path)
	
	transformed_img = transform(img)
	input_img = torch.unsqueeze(transformed_img, 0)

	orig_img = input_img.clone()
	# set model to evaluation mode and run img
	model.eval()
	output = model(input_img)
	output = F.softmax(output, dim=1)
	prediction_score, pred_label_idx = torch.topk(output, 1)
	
	print('Predicted:', pred_label_idx.item(), 'with score:', prediction_score.item())
	print('Predicted:', imagenet_labels[pred_label_idx.item()])
	
	import matplotlib.pyplot as plt
	
	# get saliency map
	saliency_map = get_saliency_image(model, pred_label_idx, input_img, "integrated_gradients")


	# plot saliency map
	plt.imshow(saliency_map[0].permute(1, 2, 0))
	# plot pixel weight bar
	plt.colorbar()
	plt.show()

	# generate mask
	masks  = generate_masks(saliency_map)
	
	# apply all masks to img
	masked_img1 = apply_mask_to_image(orig_img.clone(), masks[0])
	masked_img2 = apply_mask_to_image(orig_img.clone(), masks[1])
	masked_img3 = apply_mask_to_image(orig_img.clone(), masks[2])
	# stop time
	end = time.time()
	print("Time elapsed: ", end - start)
	orig_img = orig_img[0].permute(1, 2, 0)
	# plot masked img in shape 3*244*244
	fig, axs = plt.subplots(2, 2)
	axs[0, 0].imshow(orig_img)
	axs[0, 0].set_title('Original Image')
	axs[0, 1].imshow(masked_img1.permute(1, 2, 0))
	axs[0, 1].set_title('Masked Image 70%')
	axs[1, 0].imshow(masked_img2.permute(1, 2, 0))
	axs[1, 0].set_title('Masked Image 50%')
	axs[1, 1].imshow(masked_img3.permute(1, 2, 0))
	axs[1, 1].set_title('Masked Image 30%')





#%%
