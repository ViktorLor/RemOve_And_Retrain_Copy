# Files Explanation:

## 1. imageloading
Playground file to test image loading and basic mask operations. 
No methods used from utils. Purely for testing purposes.

## 2. food101_TrainingEvaluation
File to generate masks and evaluate the computational speed of the dataset.

## 3. utils
File containing more sophisticated methods for mask generation and special data loaders.


# Result explanation:

Memmaps are the fastest and most memory efficient way to load the dataset.
It still takes 8 minutes for 4 GB of data to be loaded.
In ImageNet this will take 800 minutes on my disk -> This is a problem.
In the paper 90 Epochs are trained. This would take 1200 hours on my disk just to load the data.