# Assignment 6

## Imports and Device Initialization

The first code block has the following - 
- Imports PyTorch modules like nn, optim, torchvision etc.
- Initializes CUDA device if GPU available else CPU

## Data Transforms
The code block 3 has the following -

- Defines train and test transforms
- Train transforms include color jitter, random crop, rotations etc. for data augmentation.
- Test transforms include normalization.

## Loading Data
The code block 4 has the following -

- Uses MNIST dataset
- Applies transforms
- Creates train and test dataloaders

## Visualize Sample Batch
Code block 6 explains the following - 

- Plots a batch of images from training set

## Model Definition
Code block 7 explains the following - 

Defines a CNN model class Net4 with the following:

- 2 Conv Blocks
- 1 Transition Block with MaxPool
- 1 Conv Block
- Global Average Pooling
- 2 Conv layers instead of FC
- Initializes the model

Prints model summary
The parameter size and other metrics are 

- Total params: 7,496
- Trainable params: 7,496
- Non-trainable params: 0
- Input size (MB): 0.00
- Forward/backward pass size (MB): 0.58
- Params size (MB): 0.03
- Estimated Total Size (MB): 0.61

## Training Loop

Code block 9 explains the following -

Defines:
- Lists to store losses and accuracies
- Dictionary to store misclassified examples
- GetCorrectPredCount function
- Train and test functions for training loop
- Main training loop

- Loops through epochs
- Trains model on train set
- Evaluates on test set
- Decreases LR using scheduler

Code block 10 runs 17 epochs to get an accuracy of 99.4 %


