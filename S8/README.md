# Batch Normalisation

The code in the file "@ERAV2/S8/Arun_Kutty_Assignment_8_BN.ipynb" implements a deep learning model using Batch Normalization (BN) and trains it on the CIFAR-10 dataset. Let's break down the code and explain its purpose, inputs, outputs, and the logic behind it.

Purpose:The main purpose of this code is to train a deep learning model using Batch Normalization (BN) and evaluate its performance on the CIFAR-10 dataset. Batch Normalization is a technique used to normalize the activations of a neural network layer, which helps in reducing the internal covariate shift and accelerating the training process.

Inputs:The code takes the CIFAR-10 dataset as input, which consists of 50,000 training images and 10,000 test images. Each image is a 32x32 color image belonging to one of 10 classes (e.g., airplane, automobile, bird, etc.).

Outputs:The code outputs the trained model's performance metrics, including training loss, training accuracy, test loss, and test accuracy for each epoch. It also displays the progress of training and the final test accuracy achieved by the model.

BN's accuracy increased when better image augmentation was performed -
train_transforms = transforms.Compose([
transforms.RandomHorizontalFlip(), # Randomly flip the images on the horizontal axis
transforms.RandomCrop(32, padding=4), # Apply random crops to the images with padding
transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Slight color jitter
transforms.RandomRotation(15), # Randomly rotate images within the specified angle range
transforms.ToTensor(), # Convert the images to PyTorch tensors
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) # Normalize the images
])

The model did not have overfitting and was able to give parameters less than 50,000 and accuracy of more than 70 %

The loss used was Cross entropy and the list of misclassified images is also shown in the notebook

# Layer Normalisation

For LN, the images werent normalised during image augmentation but more during the layers. But the observation was the size of the model increased rapidly and accuracy was much lower than BN.

In the network tried different things to keep the model parameters below 50k but with every case the accuracy did not go beyond 45%. Tried my level best !!

Used an Adam optimiser instead of SGD for faster convergence but that increased the accuracy by slight 4-5 % .
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

# Group Normalisation

For GN , the approach was the same as LN but instead of Layer normalisation used this line of code
nn.GroupNorm(num_groups, x),

The image augmentation was same like LN.

As LN the parameter size was less than 50 k but the accuracy didnt go beyond 59%
