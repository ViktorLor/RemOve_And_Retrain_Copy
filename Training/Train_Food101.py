"""
Training the ResNet-50 model on the Food-101 dataset.

Parameters used: https://github.com/google-research/google-research/blob/master/interpretability_benchmark/train_resnet.py#L113


"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torchvision import models

# define seed
torch.manual_seed(0)

# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Not enough memory on GPu, so use CPU
device = torch.device("cpu")
# Define transforms for training and validation data
transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.561, 0.440, 0.312], std=[0.252, 0.256, 0.259])
])



# Load the Food 101 dataset
train_dataset = torchvision.datasets.Food101(root='../Data', transform=transformer)
print("Loaded Dataset")
# Split the dataset into train and validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Create dataloaders for training and validation data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

# Load the pre-trained ResNet50 model
model = models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

print("Loaded Weights")
# Replace the last layer with a new fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 101)

# Send the model to the device
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.7, momentum=0.9, weight_decay=0.0001)

print("started training model")

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    # print epoch
    print("Epoch: ", epoch+1)
    for i, data in enumerate(train_loader, 0):
        # Get the inputs and labels
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

# save the model
torch.save(model.state_dict(), 'ResNet50_Food101.pth')
# test the model and the accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# save the accuracy in a file
with open('accuracy.txt', 'w') as f:
    f.write('Accuracy of the network on the validation images: %d %%' % (100 * correct / total))
