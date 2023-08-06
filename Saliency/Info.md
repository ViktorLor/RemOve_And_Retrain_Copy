# Explanations of the files:

Files to create the baseline and the images with blocked out pixels, using the saliency  ROAR method:

generate_random_baseline_mnist.py: Creates the images of the random baseline by blocking [0.1,0.3,0.5,0.7,0.9]% of pixels.

saliency_mnist.py: Creates the images by using integrated gradients and removing the most important pixels.

saliency_run.py: 3D image saliency generation

saliency_transform.py: Methods and functions for saliency_run.py