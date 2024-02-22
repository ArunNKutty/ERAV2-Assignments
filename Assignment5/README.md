# Assignment 5

This assignment is about redoing Assignment 4 in a modular way.

For this assignment we have divided the project into 3 parts.

- Session5.ipynb
- Models
- Utils

## Session5.ipynb

The \*Session5.ipynb leverages the model definition and training pipeline functions defined in utils.py and model.py to train a CNN on MNIST. The training loop, optimization, and analysis helpers from utils.py are used to actually train, evaluate and visualize performance of the Net model from model.py.

### Imports and Setup

- Imports modules from utils.py and model.py
- Sets up device, hyperparameters, data transformations, etc.

### Data Loading and Visualization

- Uses load_data and create_dataloader from utils.py to load and dataloader the MNIST dataset
- Uses plot_data from utils.py to visualize a sample batch

### Define Model

- Imports the Net model architecture from model.py
- Instantiates the model and moves it to the device

### Training and Evaluation

- Uses train, test, get_optimizer, get_scheduler and run_epochs functions from utils.py to train and evaluate the model
- Runs for 20 epochs with chosen hyperparameters
- Logs training and test accuracy/loss

### Analysis

- Uses plot_accuracy_loss to plot accuracy and loss graphs for analysis

## Utils file

The Below functions in Utils functions help in different utility areas as mentioned.

### Data Loading and Transformation

- _load_data_ - Loads the MNIST dataset, with options to specify train vs test, download dataset, and transform
- _train_data_transformation_ - Applies data augmentation transforms to the training data
- _test_data_transformation_ - Applies normalization transform to the test data

### Dataloaders

- _create_dataloader_ - Creates a PyTorch dataloader from the dataset
- _plot_data_ - Plots a sample batch of data from the dataloader for visualization

### Training Loop Helpers

- _train_ - Training loop definition
- _test_ - Test loop definition
- _GetCorrectPredCount_ - Helper function to count correct predictions
- _get_optimizer_ - Returns optimizer
- _get_scheduler_ - Returns learning rate scheduler
- _run_epochs_ - Main training loop runner

### Analysis Helpers

- _plot_accuracy_loss_ - Plots accuracy and loss graphs

# Models file

This file defines two CNN models:

- _Net_ - Basic CNN model
- _Net2_ - CNN model without bias terms

Both models consist of:

- 4 Convolutional layers
- 2 Fully connected layers
- Log softmax output layer
- Forward pass is defined to depict the output shape transformation at each layer
