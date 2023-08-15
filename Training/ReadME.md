## Files

1. Train_MNIST_SimpleCNN_original_Dataset

Trains the model on the original MNIST dataset. The model is saved in the folder models/mnist/original_net.pth

2. Train_MNIST_SimpleCNN_5times_thresholdblock

After generating the altered dataset, this file trains the model on the altered dataset. The model is saved in the folder models/mnist/{
method}/models/{threshold}_{1-5}_net.pth

3. Train_Food101_ResNet50_original_Dataset

Trains the ResNET50 model on the original dataset.



## To Know about Learning parameters:

LR= 0.1 -> 13 % accuracy
LR= 0.05 -> 27% accuracy