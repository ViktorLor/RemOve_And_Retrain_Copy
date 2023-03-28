"""
Created on Mon Nov  4 14:50:00 2019

Author: Viktor Loreth

This file is used to run the saliency_transform.py file on all images in a given path.
It is used to generate the saliency maps for the images in the ILSVRC dataset.
The files are saved in a folder called ILSVRC30, ILSVRC50, ILSVRC70 respectively.

Right now we only run it on the validation set, but it can be easily adapted to run on the training set as well.
"""
import torchvision

import saliency_transform as sal_help
import sys
import os
import torch
import numpy as np
import torch.distributed as dist
from torch.multiprocessing import Process
from PIL import Image

# run saliency_helper for all images in path

# LINUX PATH
path = r'/home/viktorl/Intepretable_AI_PR_Loreth/Dataset/food-101/images'
# Windows path:
# path = 'C:\\Users\\Vik\\Documents\\4. Private\\01. University\\2022_Sem5\\Intepretable_AI\\imagnet_samples\\imagenet1000samples'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.cuda.empty_cache()
torch.cuda.synchronize()

thresholds = [0.3, 0.5, 0.7]

model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
model.to(device)
model.eval()

config = 'ImageNet'
config = 'food101'

# create folders for the imagnet_samples
if config == 'ImageNet':
    images = os.listdir(path)
    for i in range(len(thresholds)):
        if not os.path.exists(path + '/ILSVRC' + str(int(thresholds[i] * 100))):
            os.makedirs(path + '/ILSVRC' + str(int(thresholds[i] * 100)))

    for i, image in enumerate(images):
        sal_help.calculate_saliency_map(model, image, thresholds=thresholds, cuda=use_cuda,
                                        project_path=path)
    if i % 10 == 0:
        print(i, " Image done")

if config == 'food101':
    folders = os.listdir(path)
    for z,folder in enumerate(folders):
        images = os.listdir(path + '/' + folder)
        for i in range(len(thresholds)):
            if not os.path.exists(path + '/' + folder + str(int(thresholds[i] * 100))):
                os.makedirs(path + '/' + folder + str(int(thresholds[i] * 100)))

        for i, image in enumerate(images):
            try:
                sal_help.calculate_saliency_map(model, image, thresholds=thresholds, cuda=use_cuda,
                                                project_path=path + '/' + folder)
            except:
                print("Error with image: ", image)
                #write to logfile
                with open("logfile.txt", "a") as logfile:
                    logfile.write("Error with image: " + image)

            if i%100 == 0:
                print(i, z, "image done", len(folders))
# %%
