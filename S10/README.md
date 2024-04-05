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

**Layer Structure**

```
PrepLayer
	 torch.Size([1, 64, 32, 32])

Layer 1, X
	 torch.Size([1, 128, 16, 16])

Layer 1, R1
	 torch.Size([1, 128, 16, 16])

Layer 1, X + R1
	 torch.Size([1, 128, 16, 16])

Layer 2
	 torch.Size([1, 256, 8, 8])

Layer 3, X
	 torch.Size([1, 512, 4, 4])

Layer 3, R2
	 torch.Size([1, 512, 4, 4])

Layer 3, X + R2
	 torch.Size([1, 512, 4, 4])

Max Pooling
	 torch.Size([1, 512, 1, 1])

Reshape before FC
	 torch.Size([1, 512])

After FC
	 torch.Size([1, 10])

```

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

<br>

## LR Finder

**Code**

```
# Create LR finder object
lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader=train_loader, end_lr=10, num_iter=200, start_lr=1e-2)
# https://github.com/davidtvs/pytorch-lr-finder/issues/88
plot, suggested_lr = lr_finder.plot(suggest_lr=True)
lr_finder.reset()
# plot.figure.savefig("LRFinder - Suggested Max LR.png")

```

**Suggested Max LR**

![Alt text](<../Files/LR Finder.png>)

<br>

## Training logs

```


Batch size: 512, Total epochs: 24


Epoch 1
Train: Loss=1.5520, Batch_id=97, Accuracy=29.41: 100%|██████████| 98/98 [00:21<00:00,  4.63it/s]
Test set: Average loss: 0.0033,  Accuracy: 4142/10000  (41.42%)


Epoch 2
Train: Loss=1.1888, Batch_id=97, Accuracy=49.93: 100%|██████████| 98/98 [00:20<00:00,  4.71it/s]
Test set: Average loss: 0.0024,  Accuracy: 5616/10000  (56.16%)


Epoch 3
Train: Loss=0.9522, Batch_id=97, Accuracy=62.26: 100%|██████████| 98/98 [00:20<00:00,  4.75it/s]
Test set: Average loss: 0.0020,  Accuracy: 6538/10000  (65.38%)


Epoch 4
Train: Loss=0.7499, Batch_id=97, Accuracy=70.58: 100%|██████████| 98/98 [00:21<00:00,  4.61it/s]
Test set: Average loss: 0.0017,  Accuracy: 7122/10000  (71.22%)


Epoch 5
Train: Loss=0.6569, Batch_id=97, Accuracy=76.07: 100%|██████████| 98/98 [00:20<00:00,  4.77it/s]
Test set: Average loss: 0.0013,  Accuracy: 7737/10000  (77.37%)


Epoch 6
Train: Loss=0.6123, Batch_id=97, Accuracy=79.32: 100%|██████████| 98/98 [00:21<00:00,  4.58it/s]
Test set: Average loss: 0.0013,  Accuracy: 7900/10000  (79.00%)


Epoch 7
Train: Loss=0.4997, Batch_id=97, Accuracy=81.51: 100%|██████████| 98/98 [00:21<00:00,  4.63it/s]
Test set: Average loss: 0.0010,  Accuracy: 8225/10000  (82.25%)


Epoch 8
Train: Loss=0.4568, Batch_id=97, Accuracy=83.16: 100%|██████████| 98/98 [00:21<00:00,  4.62it/s]
Test set: Average loss: 0.0010,  Accuracy: 8387/10000  (83.87%)


Epoch 9
Train: Loss=0.4182, Batch_id=97, Accuracy=84.98: 100%|██████████| 98/98 [00:20<00:00,  4.73it/s]
Test set: Average loss: 0.0010,  Accuracy: 8399/10000  (83.99%)


Epoch 10
Train: Loss=0.3464, Batch_id=97, Accuracy=86.14: 100%|██████████| 98/98 [00:21<00:00,  4.55it/s]
Test set: Average loss: 0.0009,  Accuracy: 8524/10000  (85.24%)


Epoch 11
Train: Loss=0.4877, Batch_id=97, Accuracy=87.15: 100%|██████████| 98/98 [00:20<00:00,  4.71it/s]
Test set: Average loss: 0.0009,  Accuracy: 8510/10000  (85.10%)


Epoch 12
Train: Loss=0.3593, Batch_id=97, Accuracy=88.12: 100%|██████████| 98/98 [00:21<00:00,  4.56it/s]
Test set: Average loss: 0.0009,  Accuracy: 8542/10000  (85.42%)


Epoch 13
Train: Loss=0.3405, Batch_id=97, Accuracy=88.65: 100%|██████████| 98/98 [00:21<00:00,  4.58it/s]
Test set: Average loss: 0.0009,  Accuracy: 8587/10000  (85.87%)


Epoch 14
Train: Loss=0.3532, Batch_id=97, Accuracy=89.91: 100%|██████████| 98/98 [00:21<00:00,  4.61it/s]
Test set: Average loss: 0.0008,  Accuracy: 8723/10000  (87.23%)


Epoch 15
Train: Loss=0.2154, Batch_id=97, Accuracy=90.45: 100%|██████████| 98/98 [00:20<00:00,  4.68it/s]
Test set: Average loss: 0.0008,  Accuracy: 8752/10000  (87.52%)


Epoch 16
Train: Loss=0.4153, Batch_id=97, Accuracy=90.77: 100%|██████████| 98/98 [00:21<00:00,  4.51it/s]
Test set: Average loss: 0.0007,  Accuracy: 8891/10000  (88.91%)


Epoch 17
Train: Loss=0.2890, Batch_id=97, Accuracy=91.13: 100%|██████████| 98/98 [00:21<00:00,  4.58it/s]
Test set: Average loss: 0.0007,  Accuracy: 8903/10000  (89.03%)


Epoch 18
Train: Loss=0.2290, Batch_id=97, Accuracy=92.10: 100%|██████████| 98/98 [00:21<00:00,  4.59it/s]
Test set: Average loss: 0.0007,  Accuracy: 8901/10000  (89.01%)


Epoch 19
Train: Loss=0.1481, Batch_id=97, Accuracy=92.53: 100%|██████████| 98/98 [00:21<00:00,  4.64it/s]
Test set: Average loss: 0.0007,  Accuracy: 8959/10000  (89.59%)


Epoch 20
Train: Loss=0.2575, Batch_id=97, Accuracy=93.16: 100%|██████████| 98/98 [00:20<00:00,  4.69it/s]
Test set: Average loss: 0.0007,  Accuracy: 8961/10000  (89.61%)


Epoch 21
Train: Loss=0.2115, Batch_id=97, Accuracy=93.41: 100%|██████████| 98/98 [00:20<00:00,  4.71it/s]
Test set: Average loss: 0.0008,  Accuracy: 8832/10000  (88.32%)


Epoch 22
Train: Loss=0.1986, Batch_id=97, Accuracy=93.64: 100%|██████████| 98/98 [00:22<00:00,  4.43it/s]
Test set: Average loss: 0.0008,  Accuracy: 8867/10000  (88.67%)


Epoch 23
Train: Loss=0.2486, Batch_id=97, Accuracy=93.85: 100%|██████████| 98/98 [00:21<00:00,  4.57it/s]
Test set: Average loss: 0.0008,  Accuracy: 8864/10000  (88.64%)


Epoch 24
Train: Loss=0.1808, Batch_id=97, Accuracy=94.56: 100%|██████████| 98/98 [00:21<00:00,  4.57it/s]
Test set: Average loss: 0.0007,  Accuracy: 9006/10000  (90.06%)
```

<br>

## Test and Train Metrics

![Alt text](../Files/Metrics.png)

<br>

## Misclassified Images

![Alt text](../Files/Misclassified.png)

<br>
