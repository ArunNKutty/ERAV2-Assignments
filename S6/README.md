# Assignment 6


# Part 1 

PART 1[250]: Rewrite the whole Excel sheet Download whole Excel sheet showing backpropagation. Explain each major step, and write it on GitHub. 

   - Use exactly the same values for all variables as used in the class
   - Take a screenshot, and show that screenshot in the readme file
   - The Excel file must be there for us to cross-check the image shown on readme (no image = no score)
   - Explain each major step
   - Show what happens to the error graph when you change the learning rate from [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] 
   - Upload all this to GitHub and then write all the above as part 1 of your README.md file. 
   - Submit details to S6 - Assignment QnA. 

_________________________________________________________________________


![image](https://github.com/ArunNKutty/ERAV2-Assignments/assets/4424906/75adba24-bbb4-48b3-9f40-777b964bde79)


The spreadsheet shows the calculations involved in backpropagation for a simple neural network with 1 input layer with 2 neurons, 1 hidden layer with 2 neurons and 1 output layer with 2 neurons.

Forward Pass:

There is a forward pass which is already given in the excel in terms of weights., The total error is calculated as E = E1 + E2 


Backpropagation:

Then we calculate the backpropagation at each node using partial derivatives

![image](https://github.com/ArunNKutty/ERAV2-Assignments/assets/4424906/d90a30f7-637c-4736-bc33-ce6339002ad5)


Towards the end we are calculating the total Loss graph by filling in the values 

When learning rate is 0.1

![image](https://github.com/ArunNKutty/ERAV2-Assignments/assets/4424906/535489bd-ef3c-48a7-8879-867a09b4de31)

when learning rate is 0.2

![image](https://github.com/ArunNKutty/ERAV2-Assignments/assets/4424906/4b0e09fe-6d2d-40fd-9554-03fef43fc31b)

when learning rate is 0.5

![image](https://github.com/ArunNKutty/ERAV2-Assignments/assets/4424906/4608d150-cd28-4f8a-8229-6934f719b416)

when learning rate is 0.8

![image](https://github.com/ArunNKutty/ERAV2-Assignments/assets/4424906/69aa385a-6019-419a-b582-e3cdad6569f4)


when learning rate is 1

![image](https://github.com/ArunNKutty/ERAV2-Assignments/assets/4424906/cff769bf-b2a8-4498-840a-273d226dc52d)

when learning rate is 2 

![image](https://github.com/ArunNKutty/ERAV2-Assignments/assets/4424906/e5ea9991-35fd-4b34-98e3-b78ce740e695)


As you can see when the learning rate increases it learns very fast.



So in summary, it calculates the errors, gradients and updates the weights by backpropagating the errors from output layer to input layer.




# Part 2

PART 2 [250]: We have considered many points in our last 5 lectures. Some of these we have covered directly and some indirectly. They are:

    How many layers,
    MaxPooling,
    1x1 Convolutions,
    3x3 Convolutions,
    Receptive Field,
    SoftMax,
    Learning Rate,
    Kernels and how do we decide the number of kernels?
    Batch Normalization,
    Image Normalization,
    Position of MaxPooling,
    Concept of Transition Layers,
    Position of Transition Layer,
    DropOut
    When do we introduce DropOut, or when do we know we have some overfitting
    The distance of MaxPooling from Prediction,
    The distance of Batch Normalization from Prediction,
    When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
    How do we know our network is not going well, comparatively, very early
    Batch Size, and Effects of batch size
    etc (you can add more if we missed it here)

Refer to this code: COLABLINK

    WRITE IT AGAIN SUCH THAT IT ACHIEVES
        99.4% validation accuracy
        Less than 20k Parameters
        You can use anything from above you want. 
        Less than 20 Epochs
        Have used BN, Dropout,
        (Optional): a Fully connected layer, have used GAP. 

_______________________________________________________________________________________________________________________

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


