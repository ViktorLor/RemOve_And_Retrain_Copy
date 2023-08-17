## Files

1. Train_MNIST_SimpleCNN_original_Dataset

Trains the model on the original MNIST dataset. The model is saved in the folder models/mnist/original_net.pth

2. Train_MNIST_SimpleCNN_5times_thresholdblock

After generating the altered dataset, this file trains the model on the altered dataset. The model is saved in the folder models/mnist/{
method}/models/{threshold}_{1-5}_net.pth

3. Train_Food101_ResNet50_original_Dataset

Trains the ResNET50 model on the original dataset.





# What was tried already?

- Trying different learning rates: {0.1: 13%, 0.05: 27%, 0.01:7.8%}
- Tensorboard curves

## what needs to be added?

- look deep into paper and code
- superfit one minibatch




### How to run Tensorboard:

1. SSH with port forwarding

ssh -L 16006:127.0.0.1:6006 viktorl@student2.cp.jku.at

2. Run tensorboard from ~/Intepretable_AI_Loreth/models/food101/runs_original

 tensorboard --logdir= . --host=localhost --port=6006
 
3. Start program and open in browser