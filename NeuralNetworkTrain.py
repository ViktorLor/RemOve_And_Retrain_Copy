"""
Training the ResNet-50 model on the Food-101 dataset.

Parameters used: https://github.com/google-research/google-research/blob/master/interpretability_benchmark/train_resnet.py#L113

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

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# load resNet50 model with pretrained weights from ImageNet
model = torchvision.models.resnet50(pretrained=True)
#send model togpu
model.to('cuda')
# create a transformer to normalize the data
transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.561, 0.440, 0.312], std=[0.252, 0.256, 0.259])
])

# load the dataset
MyDataSet = torchvision.datasets.Food101(root='./Data', download=False, transform=transformer)

# Split the dataset into train and test
train_size = int(0.8 * len(MyDataSet))
test_size = len(MyDataSet) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(MyDataSet, [train_size, test_size])

# Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.7, momentum=0.9, weight_decay=0.0001)
# loss function
criterion = nn.CrossEntropyLoss().to('cuda')
#send all  to gpu


# train network

train_loss =[]
train_acc = []
test_loss = []
test_acc = []


for epoch in range(1):
    for i, data in enumerate(train_loader):
        inputs, labels = data.to('cuda')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        train_acc.append(correct / total)


        print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')

# test network
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs, labels = data
        # send data to gpu
        inputs = inputs.to('cuda')
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss.append(loss.item())
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        test_acc.append(correct / total)

        if i % 100 == 0:
            print(f'Batch: {i}, Loss: {loss.item()}')

# save the model
torch.save(model.state_dict(), 'resnet50.pth')

# save the training and testing loss and accuracy as a singular dictionary
data = {'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss, 'test_acc': test_acc}
np.save('data.npy', data)

# create plots of the training and testing loss and accuracy
import matplotlib.pyplot as plt

data = np.load('data.npy', allow_pickle=True).item()

plt.plot(data['train_loss'], label='train loss')
plt.plot(data['test_loss'], label='test loss')
plt.legend()
plt.show()

plt.plot(data['train_acc'], label='train acc')
plt.plot(data['test_acc'], label='test acc')
plt.legend()
plt.show()


# save the plots
plt.savefig('loss.png')
plt.savefig('acc.png')


