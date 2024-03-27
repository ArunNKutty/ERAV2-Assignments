## Assignment 9 Objectives

Write a new network that

- Has the architecture to C1C2C3C40 (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200 pts extra!)
- Total RF must be more than 44
- A Layer must use Depthwise Separable Convolution
- A Layers must use Dilated Convolution
- Use GAP (compulsory)
- Add FC after GAP to target #of classes (optional)
- Use Albumentation library and apply:
  - horizontal flip
  - shiftScaleRotate
  - coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
- Achieve 85% accuracy, as many epochs as you want
- Total Params to be less than 200k.
- Follows code-modularity (else 0 for full assignment)

<br>

## Dataset Details

The CIFAR10 dataset is a collection of 60,000 32x32 color images, divided into 50,000 training images and 10,000 test images. The dataset contains 10 classes, each with 6,000 images: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

<br>

## Code Overview

We explore various Normalization techniques using Convolution neural networks on CIFAR10 data. To run the code, download the Notebook and modules. Then just run the Notebook and other modules will be automatically imported. The code is structured in a modular way as below:

- **Modules**
  - [dataset.py](dataset.py)
    - Function to download and split CIFAR10 data to test and train - `split_cifar_data()`
    - Function that creates the required test and train transforms compatible with Albumentations - `apply_cifar_image_transformations()`
    - Class that applies the required transforms to dataset - CIFAR10Transforms()
    - Function to calculate mean and standard deviation of the data to normalize tensors - `calculate_mean_std()`
  - [model.py](model.py)
    - Train and test the model given the optimizer and criterion - `train_model()`, `test_model()`
    - A class called Assignment9 which implements above specified neural network
  - [utils.py](utils.py)
    - Function that detects and returns correct device including GPU and CPU - `get_device()`
    - Given a set of predictions and labels, return the cumulative correct count - `get_correct_prediction_count()`
    - Function to save model, epoch, optimizer, scheduler, loss and batch size - `save_model()`
  - [visualize.py](visualize.py)
    - Given a normalize image along with mean and standard deviation for each channels, convert it back - `convert_back_image()`
    - Plot sample training images along with the labels - `plot_sample_training_images()`
    - Plot train and test metrics - `plot_train_test_metrics()`
    - Plot incorrectly classified images along with ground truth and predicted classes - `plot_misclassified_images()`
- **[Notebook](<ERA V1 - Viraj - Assignment 09.ipynb>)**
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
    - Start training and compute various train and test metrics, save best model after each epoch
    - Plot accuracy and loss metrics, also print them in a human readable format
    - Save model after final epoch
    - Show incorrectly predicted images along with actual and predicted labels

<br>

### Parameters

```
# Output size after each block
torch.Size([2, 12, 32, 32])
torch.Size([2, 24, 16, 16])
torch.Size([2, 32, 16, 16])
torch.Size([2, 64, 14, 14])
torch.Size([2, 64, 1, 1])
torch.Size([2, 64])
torch.Size([2, 10])

# Model Summary
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 32, 32]             216
              ReLU-2            [-1, 8, 32, 32]               0
           Dropout-3            [-1, 8, 32, 32]               0
       BatchNorm2d-4            [-1, 8, 32, 32]              16
            Conv2d-5            [-1, 8, 32, 32]             576
              ReLU-6            [-1, 8, 32, 32]               0
           Dropout-7            [-1, 8, 32, 32]               0
       BatchNorm2d-8            [-1, 8, 32, 32]              16
            Conv2d-9           [-1, 12, 32, 32]             864
             ReLU-10           [-1, 12, 32, 32]               0
          Dropout-11           [-1, 12, 32, 32]               0
      BatchNorm2d-12           [-1, 12, 32, 32]              24
           Conv2d-13           [-1, 16, 32, 32]           1,728
             ReLU-14           [-1, 16, 32, 32]               0
          Dropout-15           [-1, 16, 32, 32]               0
      BatchNorm2d-16           [-1, 16, 32, 32]              32
           Conv2d-17           [-1, 32, 32, 32]           4,608
             ReLU-18           [-1, 32, 32, 32]               0
          Dropout-19           [-1, 32, 32, 32]               0
      BatchNorm2d-20           [-1, 32, 32, 32]              64
           Conv2d-21           [-1, 24, 16, 16]           6,912
             ReLU-22           [-1, 24, 16, 16]               0
          Dropout-23           [-1, 24, 16, 16]               0
      BatchNorm2d-24           [-1, 24, 16, 16]              48
           Conv2d-25           [-1, 32, 16, 16]           6,912
             ReLU-26           [-1, 32, 16, 16]               0
          Dropout-27           [-1, 32, 16, 16]               0
      BatchNorm2d-28           [-1, 32, 16, 16]              64
           Conv2d-29          [-1, 128, 16, 16]           1,152
             ReLU-30          [-1, 128, 16, 16]               0
          Dropout-31          [-1, 128, 16, 16]               0
      BatchNorm2d-32          [-1, 128, 16, 16]             256
           Conv2d-33           [-1, 32, 16, 16]          36,864
             ReLU-34           [-1, 32, 16, 16]               0
          Dropout-35           [-1, 32, 16, 16]               0
      BatchNorm2d-36           [-1, 32, 16, 16]              64
           Conv2d-37           [-1, 64, 14, 14]          18,432
             ReLU-38           [-1, 64, 14, 14]               0
          Dropout-39           [-1, 64, 14, 14]               0
      BatchNorm2d-40           [-1, 64, 14, 14]             128
           Conv2d-41           [-1, 96, 14, 14]          55,296
             ReLU-42           [-1, 96, 14, 14]               0
          Dropout-43           [-1, 96, 14, 14]               0
      BatchNorm2d-44           [-1, 96, 14, 14]             192
           Conv2d-45           [-1, 64, 14, 14]          55,296
             ReLU-46           [-1, 64, 14, 14]               0
          Dropout-47           [-1, 64, 14, 14]               0
      BatchNorm2d-48           [-1, 64, 14, 14]             128
AdaptiveAvgPool2d-49             [-1, 64, 1, 1]               0
           Linear-50                   [-1, 32]           2,080
           Linear-51                   [-1, 10]             330
================================================================
Total params: 192,298
Trainable params: 192,298
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 5.40
Params size (MB): 0.73
Estimated Total Size (MB): 6.15
----------------------------------------------------------------
```

<br>

## Training logs

```
Batch size: 128, Total epochs: 50


Epoch 1
Train: Loss=1.5597, Batch_id=390, Accuracy=31.35: 100%|██████████| 391/391 [00:18<00:00, 21.66it/s]
Test set: Average loss: 1.4526, Accuracy: 4500/10000 (45.00%)
Saving the model as best test accuracy till now is achieved!


Epoch 2
Train: Loss=1.2169, Batch_id=390, Accuracy=46.21: 100%|██████████| 391/391 [00:19<00:00, 20.34it/s]
Test set: Average loss: 1.1890, Accuracy: 5625/10000 (56.25%)
Saving the model as best test accuracy till now is achieved!


Epoch 3
Train: Loss=1.4035, Batch_id=390, Accuracy=53.05: 100%|██████████| 391/391 [00:18<00:00, 21.67it/s]
Test set: Average loss: 1.0611, Accuracy: 6156/10000 (61.56%)
Saving the model as best test accuracy till now is achieved!


Epoch 4
Train: Loss=1.1002, Batch_id=390, Accuracy=57.50: 100%|██████████| 391/391 [00:18<00:00, 21.38it/s]
Test set: Average loss: 0.9679, Accuracy: 6490/10000 (64.90%)
Saving the model as best test accuracy till now is achieved!


Epoch 5
Train: Loss=0.9335, Batch_id=390, Accuracy=59.93: 100%|██████████| 391/391 [00:18<00:00, 21.44it/s]
Test set: Average loss: 0.8628, Accuracy: 6901/10000 (69.01%)
Saving the model as best test accuracy till now is achieved!


Epoch 6
Train: Loss=1.0514, Batch_id=390, Accuracy=62.66: 100%|██████████| 391/391 [00:19<00:00, 20.26it/s]
Test set: Average loss: 0.8053, Accuracy: 7148/10000 (71.48%)
Saving the model as best test accuracy till now is achieved!


Epoch 7
Train: Loss=1.0892, Batch_id=390, Accuracy=64.87: 100%|██████████| 391/391 [00:18<00:00, 21.18it/s]
Test set: Average loss: 0.7648, Accuracy: 7350/10000 (73.50%)
Saving the model as best test accuracy till now is achieved!


Epoch 8
Train: Loss=0.9545, Batch_id=390, Accuracy=65.97: 100%|██████████| 391/391 [00:17<00:00, 22.25it/s]
Test set: Average loss: 0.7306, Accuracy: 7434/10000 (74.34%)
Saving the model as best test accuracy till now is achieved!


Epoch 9
Train: Loss=0.8904, Batch_id=390, Accuracy=67.65: 100%|██████████| 391/391 [00:19<00:00, 20.38it/s]
Test set: Average loss: 0.7105, Accuracy: 7572/10000 (75.72%)
Saving the model as best test accuracy till now is achieved!


Epoch 10
Train: Loss=0.9085, Batch_id=390, Accuracy=68.46: 100%|██████████| 391/391 [00:17<00:00, 22.37it/s]
Test set: Average loss: 0.6643, Accuracy: 7662/10000 (76.62%)
Saving the model as best test accuracy till now is achieved!


Epoch 11
Train: Loss=0.7089, Batch_id=390, Accuracy=69.72: 100%|██████████| 391/391 [00:17<00:00, 21.88it/s]
Test set: Average loss: 0.6653, Accuracy: 7744/10000 (77.44%)
Saving the model as best test accuracy till now is achieved!


Epoch 12
Train: Loss=0.9395, Batch_id=390, Accuracy=70.12: 100%|██████████| 391/391 [00:17<00:00, 22.41it/s]
Test set: Average loss: 0.6278, Accuracy: 7815/10000 (78.15%)
Saving the model as best test accuracy till now is achieved!


Epoch 13
Train: Loss=0.8052, Batch_id=390, Accuracy=70.67: 100%|██████████| 391/391 [00:17<00:00, 21.98it/s]
Test set: Average loss: 0.6024, Accuracy: 7891/10000 (78.91%)
Saving the model as best test accuracy till now is achieved!


Epoch 14
Train: Loss=0.5834, Batch_id=390, Accuracy=71.54: 100%|██████████| 391/391 [00:19<00:00, 20.09it/s]
Test set: Average loss: 0.6056, Accuracy: 7895/10000 (78.95%)
Saving the model as best test accuracy till now is achieved!


Epoch 15
Train: Loss=0.6659, Batch_id=390, Accuracy=72.09: 100%|██████████| 391/391 [00:17<00:00, 21.79it/s]
Test set: Average loss: 0.5692, Accuracy: 8030/10000 (80.30%)
Saving the model as best test accuracy till now is achieved!


Epoch 16
Train: Loss=0.7767, Batch_id=390, Accuracy=72.68: 100%|██████████| 391/391 [00:20<00:00, 19.35it/s]
Test set: Average loss: 0.5579, Accuracy: 8067/10000 (80.67%)
Saving the model as best test accuracy till now is achieved!


Epoch 17
Train: Loss=1.1557, Batch_id=390, Accuracy=73.00: 100%|██████████| 391/391 [00:17<00:00, 22.13it/s]
Test set: Average loss: 0.5553, Accuracy: 8102/10000 (81.02%)
Saving the model as best test accuracy till now is achieved!


Epoch 18
Train: Loss=0.8029, Batch_id=390, Accuracy=73.31: 100%|██████████| 391/391 [00:17<00:00, 22.04it/s]
Test set: Average loss: 0.5481, Accuracy: 8139/10000 (81.39%)
Saving the model as best test accuracy till now is achieved!


Epoch 19
Train: Loss=1.0237, Batch_id=390, Accuracy=73.99: 100%|██████████| 391/391 [00:18<00:00, 20.97it/s]
Test set: Average loss: 0.5240, Accuracy: 8177/10000 (81.77%)
Saving the model as best test accuracy till now is achieved!


Epoch 20
Train: Loss=0.7743, Batch_id=390, Accuracy=74.17: 100%|██████████| 391/391 [00:17<00:00, 22.01it/s]
Test set: Average loss: 0.5315, Accuracy: 8199/10000 (81.99%)
Saving the model as best test accuracy till now is achieved!


Epoch 21
Train: Loss=0.7171, Batch_id=390, Accuracy=74.81: 100%|██████████| 391/391 [00:19<00:00, 20.27it/s]
Test set: Average loss: 0.5237, Accuracy: 8199/10000 (81.99%)
Saving the model as best test accuracy till now is achieved!


Epoch 22
Train: Loss=0.7938, Batch_id=390, Accuracy=74.77: 100%|██████████| 391/391 [00:17<00:00, 21.97it/s]
Test set: Average loss: 0.5080, Accuracy: 8276/10000 (82.76%)
Saving the model as best test accuracy till now is achieved!


Epoch 23
Train: Loss=0.7894, Batch_id=390, Accuracy=75.20: 100%|██████████| 391/391 [00:18<00:00, 21.59it/s]
Test set: Average loss: 0.5058, Accuracy: 8270/10000 (82.70%)


Epoch 24
Train: Loss=0.6595, Batch_id=390, Accuracy=75.50: 100%|██████████| 391/391 [00:17<00:00, 22.03it/s]
Test set: Average loss: 0.5021, Accuracy: 8296/10000 (82.96%)
Saving the model as best test accuracy till now is achieved!


Epoch 25
Train: Loss=0.7422, Batch_id=390, Accuracy=75.61: 100%|██████████| 391/391 [00:19<00:00, 20.20it/s]
Test set: Average loss: 0.5182, Accuracy: 8236/10000 (82.36%)


Epoch 26
Train: Loss=0.6681, Batch_id=390, Accuracy=75.94: 100%|██████████| 391/391 [00:17<00:00, 21.98it/s]
Test set: Average loss: 0.4924, Accuracy: 8343/10000 (83.43%)
Saving the model as best test accuracy till now is achieved!


Epoch 27
Train: Loss=0.6314, Batch_id=390, Accuracy=76.22: 100%|██████████| 391/391 [00:17<00:00, 21.86it/s]
Test set: Average loss: 0.4885, Accuracy: 8319/10000 (83.19%)


Epoch 28
Train: Loss=0.7291, Batch_id=390, Accuracy=76.12: 100%|██████████| 391/391 [00:19<00:00, 20.15it/s]
Test set: Average loss: 0.4778, Accuracy: 8381/10000 (83.81%)
Saving the model as best test accuracy till now is achieved!


Epoch 29
Train: Loss=0.6909, Batch_id=390, Accuracy=76.48: 100%|██████████| 391/391 [00:17<00:00, 22.06it/s]
Test set: Average loss: 0.4737, Accuracy: 8399/10000 (83.99%)
Saving the model as best test accuracy till now is achieved!


Epoch 30
Train: Loss=0.8150, Batch_id=390, Accuracy=76.63: 100%|██████████| 391/391 [00:18<00:00, 20.94it/s]
Test set: Average loss: 0.4622, Accuracy: 8440/10000 (84.40%)
Saving the model as best test accuracy till now is achieved!


Epoch 31
Train: Loss=0.6408, Batch_id=390, Accuracy=76.69: 100%|██████████| 391/391 [00:17<00:00, 22.18it/s]
Test set: Average loss: 0.4727, Accuracy: 8418/10000 (84.18%)


Epoch 32
Train: Loss=0.9029, Batch_id=390, Accuracy=76.94: 100%|██████████| 391/391 [00:18<00:00, 21.32it/s]
Test set: Average loss: 0.4735, Accuracy: 8400/10000 (84.00%)


Epoch 33
Train: Loss=0.7119, Batch_id=390, Accuracy=77.10: 100%|██████████| 391/391 [00:18<00:00, 21.06it/s]
Test set: Average loss: 0.4582, Accuracy: 8457/10000 (84.57%)
Saving the model as best test accuracy till now is achieved!


Epoch 34
Train: Loss=0.5997, Batch_id=390, Accuracy=77.32: 100%|██████████| 391/391 [00:17<00:00, 22.12it/s]
Test set: Average loss: 0.4629, Accuracy: 8445/10000 (84.45%)


Epoch 35
Train: Loss=0.8371, Batch_id=390, Accuracy=77.37: 100%|██████████| 391/391 [00:21<00:00, 18.29it/s]
Test set: Average loss: 0.4611, Accuracy: 8410/10000 (84.10%)


Epoch 36
Train: Loss=0.6587, Batch_id=390, Accuracy=77.72: 100%|██████████| 391/391 [00:18<00:00, 21.30it/s]
Test set: Average loss: 0.4482, Accuracy: 8461/10000 (84.61%)
Saving the model as best test accuracy till now is achieved!


Epoch 37
Train: Loss=0.5546, Batch_id=390, Accuracy=77.69: 100%|██████████| 391/391 [00:18<00:00, 21.28it/s]
Test set: Average loss: 0.4396, Accuracy: 8531/10000 (85.31%)
Saving the model as best test accuracy till now is achieved!


Epoch 38
Train: Loss=0.5891, Batch_id=390, Accuracy=77.72: 100%|██████████| 391/391 [00:18<00:00, 21.71it/s]
Test set: Average loss: 0.4411, Accuracy: 8527/10000 (85.27%)


Epoch 39
Train: Loss=0.4353, Batch_id=390, Accuracy=78.04: 100%|██████████| 391/391 [00:17<00:00, 22.08it/s]
Test set: Average loss: 0.4384, Accuracy: 8520/10000 (85.20%)


Epoch 40
Train: Loss=0.5473, Batch_id=390, Accuracy=77.93: 100%|██████████| 391/391 [00:18<00:00, 20.99it/s]
Test set: Average loss: 0.4431, Accuracy: 8519/10000 (85.19%)


Epoch 41
Train: Loss=0.6322, Batch_id=390, Accuracy=78.36: 100%|██████████| 391/391 [00:17<00:00, 21.82it/s]
Test set: Average loss: 0.4303, Accuracy: 8536/10000 (85.36%)
Saving the model as best test accuracy till now is achieved!


Epoch 42
Train: Loss=0.6371, Batch_id=390, Accuracy=78.56: 100%|██████████| 391/391 [00:19<00:00, 20.04it/s]
Test set: Average loss: 0.4386, Accuracy: 8506/10000 (85.06%)


Epoch 43
Train: Loss=0.5805, Batch_id=390, Accuracy=78.52: 100%|██████████| 391/391 [00:17<00:00, 21.84it/s]
Test set: Average loss: 0.4211, Accuracy: 8559/10000 (85.59%)
Saving the model as best test accuracy till now is achieved!


Epoch 44
Train: Loss=0.4752, Batch_id=390, Accuracy=78.65: 100%|██████████| 391/391 [00:21<00:00, 18.35it/s]
Test set: Average loss: 0.4355, Accuracy: 8562/10000 (85.62%)
Saving the model as best test accuracy till now is achieved!


Epoch 45
Train: Loss=0.6549, Batch_id=390, Accuracy=78.65: 100%|██████████| 391/391 [00:17<00:00, 21.94it/s]
Test set: Average loss: 0.4277, Accuracy: 8554/10000 (85.54%)


Epoch 46
Train: Loss=0.6392, Batch_id=390, Accuracy=78.91: 100%|██████████| 391/391 [00:18<00:00, 20.73it/s]
Test set: Average loss: 0.4219, Accuracy: 8579/10000 (85.79%)
Saving the model as best test accuracy till now is achieved!


Epoch 47
Train: Loss=0.5987, Batch_id=390, Accuracy=79.14: 100%|██████████| 391/391 [00:17<00:00, 21.92it/s]
Test set: Average loss: 0.4081, Accuracy: 8636/10000 (86.36%)
Saving the model as best test accuracy till now is achieved!


Epoch 48
Train: Loss=0.5647, Batch_id=390, Accuracy=79.20: 100%|██████████| 391/391 [00:18<00:00, 21.63it/s]
Test set: Average loss: 0.4262, Accuracy: 8561/10000 (85.61%)


Epoch 49
Train: Loss=0.6926, Batch_id=390, Accuracy=79.12: 100%|██████████| 391/391 [00:18<00:00, 20.95it/s]
Test set: Average loss: 0.4146, Accuracy: 8588/10000 (85.88%)


Epoch 50
Train: Loss=0.6054, Batch_id=390, Accuracy=79.16: 100%|██████████| 391/391 [00:18<00:00, 21.26it/s]
Test set: Average loss: 0.4229, Accuracy: 8556/10000 (85.56%)
```

## Findings & Observations

- Model Notes
  - Stride = 2 implemented in last layer of block 2
  - Depthwise separable convolution implemented in block 3
  - Dilated convolution implemented in block 4
  - Global Average Pooling implemented after block 4
  - Output block has fully connected layers after GAP
- Paramter count: 192,298 (Target 200,000)
- **_Total number of epochs is 50 and >85% test accuracy was consistently achieved from Epoch 37_**
- **_Highest test accuracy is 86.3600% and highest train accuracy is 79.1600%_**
- Training is being done in a harder way with cutout etc. to ensure that the model does not overfit and learns well

<br>
