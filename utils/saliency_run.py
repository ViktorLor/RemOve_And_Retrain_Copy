"""
Created on Mon Nov  4 14:50:00 2019

Author: Viktor Loreth

This file is used to run the saliency_helper_by_vlo.py file on all images in a given path.
It is used to generate the saliency maps for the images in the ILSVRC dataset.
The files are saved in a folder called ILSVRC30, ILSVRC50, ILSVRC70 respectively.

Right now we only run it on the validation set, but it can be easily adapted to run on the training set as well.
"""
import torchvision

import saliency_helper_by_vlo as sal_help
import sys
import os
import torch
import numpy as np
import torch.distributed as dist
from torch.multiprocessing import Process

# run saliency_helper for all images in path

# LINUX PATH
#path = r'/home/viktorl/Intepretable_AI_PR_Loreth/Dataset/ILSVRC/Data/CLS-LOC/val'
#Windows path:
path = 'C:\\Users\\Vik\\Documents\\4. Private\\01. University\\2022_Sem5\\Intepretable_AI\\datasets\\imagenet1000samples'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
world_size = 1

torch.cuda.empty_cache()
torch.cuda.synchronize()

thresholds = [0.3, 0.5, 0.7]

model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
model.to(device)
model.eval()
images = os.listdir(path)

for image in images:
    masks, masks_len = sal_help.calculate_saliency_map(model, path + '/' + image, thresholds=thresholds)

    # if dir does not exist, create it
    for i in range(len(masks)):
        if not os.path.exists(path + '/ILSVRC' + str(int(thresholds[i]*100))):
            os.makedirs(path + '/ILSVRC' + str(int(thresholds[i]*100)))


    for i in range(len(masks)):
        # convert mask to np array
        masks[i] = masks[i].numpy()
        # save masks
        np.save(path + '\\ILSVRC' + str(int(thresholds[i]*100)) + '\\' + image[:-5], masks[i])
    print("Image done")


#%%
