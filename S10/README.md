# [Assignment 10]

## Assignment Objectives

Write a new network that has

- [x] Target Accuracy: 90% **(Final Test Accuracy - 90.06%)**
- [x] Total Epochs = 24
- [x] Uses One Cycle Policy with no annihilation, Max at Epoch 5, Uses LR Finder for Max
- [x] Batch size = 512
- [x] Use ADAM, and CrossEntropyLoss

<br>

## Code Overview

To run the code, download the Notebook. Then just run the Notebook and other modules will be automatically imported. The code is structured in a modular way as below:

- **Modules**
  - [dataset.py](modules/dataset.py)
    - Function to download and split CIFAR10 data to test and train - `split_cifar_data()`
    - Function that creates the required test and train transforms compatible with Albumentations - `apply_cifar_image_transformations()`
    - Class that applies the required transforms to dataset - CIFAR10Transforms()
    - Function to calculate mean and standard deviation of the data to normalize tensors - `calculate_mean_std()`
  - [custom_resnet.py](modules/custom_resnet.py)
    - A class called CustomResNet which implements above specified neural network
    - Detailed model summary - `detailed_model_summary()`
  - [trainer.py](modules/trainer.py)
    - Train and test the model given the optimizer and criterion - `train_model()`, `test_model()`, `train_and_test_model()`
  - [utils.py](modules/utils.py)
    - Function that detects and returns correct device including GPU and CPU - `get_device()`
    - Given a set of predictions and labels, return the cumulative correct count - `get_correct_prediction_count()`
    - Function to save model, epoch, optimizer, scheduler, loss and batch size - `save_model()`
    - Pretty print training log - `pretty_print_metrics()`
  - [visualize.py](modules/visualize.py)
    - Given a normalize image along with mean and standard deviation for each channels, convert it back - `convert_back_image()`
    - Plot sample training images along with the labels - `plot_sample_training_images()`
    - Plot train and test metrics - `plot_train_test_metrics()`
    - Plot incorrectly classified images along with ground truth and predicted classes - `plot_misclassified_images()`
- **[Notebook](<ERA V1 - Viraj - Assignment 10.ipynb>)**
  - **Flow**
    - Install and import required libraries
    - Mount Google drive which contains our modules and import them
    - Get device and dataset statistics
    - Apply test and train transformations
    - Split the data to test and train after downloading and applying Transformations
    - Specify the data loader depending on architecture and batch size
    - Define the class labels in a human readable format
    - Display sample images from the training data post transformations
    - Load model to device
    - Show model summary along with tensor size after each block
    - Use LR finder and Once cycle policy
    - Start training and compute various train and test metrics, save best model after each epoch
    - Plot accuracy and loss metrics, also print them in a human readable format
    - Save model after final epoch
    - Show incorrectly predicted images along with actual and predicted labels

<br>

## Model Parameters
**Parameters**

```
========================================================================================================================
Layer (type:depth-idx)                   Input Shape      Kernel Shape     Output Shape     Param #          Trainable
========================================================================================================================
CustomResNet                             [1, 3, 32, 32]   --               [1, 10]          --               True
├─Sequential: 1-1                        [1, 3, 32, 32]   --               [1, 64, 32, 32]  --               True
│    └─Conv2d: 2-1                       [1, 3, 32, 32]   [3, 3]           [1, 64, 32, 32]  1,728            True
│    └─BatchNorm2d: 2-2                  [1, 64, 32, 32]  --               [1, 64, 32, 32]  128              True
│    └─ReLU: 2-3                         [1, 64, 32, 32]  --               [1, 64, 32, 32]  --               --
│    └─Dropout: 2-4                      [1, 64, 32, 32]  --               [1, 64, 32, 32]  --               --
├─Sequential: 1-2                        [1, 64, 32, 32]  --               [1, 128, 16, 16] --               True
│    └─Conv2d: 2-5                       [1, 64, 32, 32]  [3, 3]           [1, 128, 32, 32] 73,728           True
│    └─MaxPool2d: 2-6                    [1, 128, 32, 32] 2                [1, 128, 16, 16] --               --
│    └─BatchNorm2d: 2-7                  [1, 128, 16, 16] --               [1, 128, 16, 16] 256              True
│    └─ReLU: 2-8                         [1, 128, 16, 16] --               [1, 128, 16, 16] --               --
│    └─Dropout: 2-9                      [1, 128, 16, 16] --               [1, 128, 16, 16] --               --
├─Sequential: 1-3                        [1, 128, 16, 16] --               [1, 128, 16, 16] --               True
│    └─Conv2d: 2-10                      [1, 128, 16, 16] [3, 3]           [1, 128, 16, 16] 147,456          True
│    └─BatchNorm2d: 2-11                 [1, 128, 16, 16] --               [1, 128, 16, 16] 256              True
│    └─ReLU: 2-12                        [1, 128, 16, 16] --               [1, 128, 16, 16] --               --
│    └─Dropout: 2-13                     [1, 128, 16, 16] --               [1, 128, 16, 16] --               --
│    └─Conv2d: 2-14                      [1, 128, 16, 16] [3, 3]           [1, 128, 16, 16] 147,456          True
│    └─BatchNorm2d: 2-15                 [1, 128, 16, 16] --               [1, 128, 16, 16] 256              True
│    └─ReLU: 2-16                        [1, 128, 16, 16] --               [1, 128, 16, 16] --               --
│    └─Dropout: 2-17                     [1, 128, 16, 16] --               [1, 128, 16, 16] --               --
├─Sequential: 1-4                        [1, 128, 16, 16] --               [1, 256, 8, 8]   --               True
│    └─Conv2d: 2-18                      [1, 128, 16, 16] [3, 3]           [1, 256, 16, 16] 294,912          True
│    └─MaxPool2d: 2-19                   [1, 256, 16, 16] 2                [1, 256, 8, 8]   --               --
│    └─BatchNorm2d: 2-20                 [1, 256, 8, 8]   --               [1, 256, 8, 8]   512              True
│    └─ReLU: 2-21                        [1, 256, 8, 8]   --               [1, 256, 8, 8]   --               --
│    └─Dropout: 2-22                     [1, 256, 8, 8]   --               [1, 256, 8, 8]   --               --
├─Sequential: 1-5                        [1, 256, 8, 8]   --               [1, 512, 4, 4]   --               True
│    └─Conv2d: 2-23                      [1, 256, 8, 8]   [3, 3]           [1, 512, 8, 8]   1,179,648        True
│    └─MaxPool2d: 2-24                   [1, 512, 8, 8]   2                [1, 512, 4, 4]   --               --
│    └─BatchNorm2d: 2-25                 [1, 512, 4, 4]   --               [1, 512, 4, 4]   1,024            True
│    └─ReLU: 2-26                        [1, 512, 4, 4]   --               [1, 512, 4, 4]   --               --
│    └─Dropout: 2-27                     [1, 512, 4, 4]   --               [1, 512, 4, 4]   --               --
├─Sequential: 1-6                        [1, 512, 4, 4]   --               [1, 512, 4, 4]   --               True
│    └─Conv2d: 2-28                      [1, 512, 4, 4]   [3, 3]           [1, 512, 4, 4]   2,359,296        True
│    └─BatchNorm2d: 2-29                 [1, 512, 4, 4]   --               [1, 512, 4, 4]   1,024            True
│    └─ReLU: 2-30                        [1, 512, 4, 4]   --               [1, 512, 4, 4]   --               --
│    └─Dropout: 2-31                     [1, 512, 4, 4]   --               [1, 512, 4, 4]   --               --
│    └─Conv2d: 2-32                      [1, 512, 4, 4]   [3, 3]           [1, 512, 4, 4]   2,359,296        True
│    └─BatchNorm2d: 2-33                 [1, 512, 4, 4]   --               [1, 512, 4, 4]   1,024            True
│    └─ReLU: 2-34                        [1, 512, 4, 4]   --               [1, 512, 4, 4]   --               --
│    └─Dropout: 2-35                     [1, 512, 4, 4]   --               [1, 512, 4, 4]   --               --
├─MaxPool2d: 1-7                         [1, 512, 4, 4]   4                [1, 512, 1, 1]   --               --
├─Linear: 1-8                            [1, 512]         --               [1, 10]          5,130            True
========================================================================================================================
Total params: 6,573,130
Trainable params: 6,573,130
Non-trainable params: 0
Total mult-adds (M): 379.27
========================================================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 4.65
Params size (MB): 26.29
Estimated Total Size (MB): 30.96
========================================================================================================================

```

## Training logs

![image](https://github.com/ArunNKutty/ERAV2-Assignments/assets/4424906/be1632ca-728c-48c2-b7f8-97573e624243)


