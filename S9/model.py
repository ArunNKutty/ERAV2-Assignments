"""Module to define the model and train and test functions."""

# from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import get_correct_prediction_count


############# Train and Test Functions #############


def train_model(
    model, device, train_loader, optimizer, criterion, train_acc, train_losses
):
    """
    Function to train the model on the train dataset.
    """

    # Initialize the model to train mode
    model.train()

    # Initialize progress bar
    pbar = tqdm(train_loader)

    # Reset the loss and correct predictions for the epoch
    train_loss = 0
    correct = 0
    processed = 0

    # Iterate over the train loader
    for batch_idx, (data, target) in enumerate(pbar):
        # Move data and labels to device
        data, target = data.to(device), target.to(device)
        # Clear the gradients for the optimizer to avoid accumulation
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss for the batch
        loss = criterion(pred, target)
        # Update the loss
        train_loss += loss.item()

        # Backpropagation to calculate the gradients
        loss.backward()
        # Update the weights
        optimizer.step()

        # Get the count of correct predictions
        correct += get_correct_prediction_count(pred, target)
        processed += len(data)

        # Update the progress bar
        msg = f"Train: Loss={loss.item():0.4f}, Batch_id={batch_idx}, Accuracy={100*correct/processed:0.2f}"
        pbar.set_description(desc=msg)

    # Close the progress bar
    pbar.close()

    # Append the final loss and accuracy for the epoch
    train_acc.append(100 * correct / processed)
    train_losses.append(train_loss / len(train_loader))


def test_model(
    model,
    device,
    test_loader,
    criterion,
    test_acc,
    test_losses,
    misclassified_image_data,
):
    """
    Function to test the model on the test dataset.
    """

    # Initialize the model to evaluation mode
    model.eval()

    # Reset the loss and correct predictions for the epoch
    test_loss = 0
    correct = 0

    # Disable gradient calculation while testing
    with torch.no_grad():
        for data, target in test_loader:
            # Move data and labels to device
            data, target = data.to(device), target.to(device)

            # Predict using model
            output = model(data)
            # Calculate loss for the batch
            test_loss += criterion(output, target, reduction="sum").item()

            # Get the index of the max log-probability
            pred = output.argmax(dim=1)
            # Check if the prediction is correct
            correct_mask = pred.eq(target)
            # Save the incorrect predictions
            incorrect_indices = ~correct_mask

            # Store images incorrectly predicted, generated predictions and the actual value
            misclassified_image_data["images"].extend(data[incorrect_indices])
            misclassified_image_data["ground_truths"].extend(target[incorrect_indices])
            misclassified_image_data["predicted_vals"].extend(pred[incorrect_indices])

            # Get the count of correct predictions
            correct += get_correct_prediction_count(output, target)

    # Calculate the final loss
    test_loss /= len(test_loader.dataset)
    # Append the final loss and accuracy for the epoch
    test_acc.append(100.0 * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    # Print the final test loss and accuracy
    print(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.2f}%)"
    )


############# Assignment 9 Model #############


class Assignment9(nn.Module):
    """This defines the structure of the NN."""

    # Class variable to print shape
    print_shape = False
    # Default dropout value
    dropout_value = 0.05

    def __init__(self):
        super().__init__()

        #  Model Notes
        # Stride = 2 implemented in last layer of block 2
        # Depthwise separable convolution implemented in block 3
        # Dilated convolution implemented in block 4
        # Global Average Pooling implemented after block 4
        # Output block has fully connected layers

        self.block1 = nn.Sequential(
            # Layer 1
            nn.Conv2d(
                in_channels=3,
                out_channels=8,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(8),
            # Layer 2
            nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(8),
            # Layer 3
            nn.Conv2d(
                in_channels=8,
                out_channels=12,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(12),
        )

        self.block2 = nn.Sequential(
            # Layer 1
            nn.Conv2d(
                in_channels=12,
                out_channels=16,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(16),
            # Layer 2
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(32),
            # Layer 3
            nn.Conv2d(
                in_channels=32,
                out_channels=24,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(24),
        )

        self.block3 = nn.Sequential(
            # Layer 1
            nn.Conv2d(
                in_channels=24,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(32),
            ##################### Depthwise Convolution #####################
            # Layer 2
            nn.Conv2d(
                in_channels=32,
                out_channels=128,
                kernel_size=(3, 3),
                groups=32,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(128),
            # Layer 3
            nn.Conv2d(
                in_channels=128,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(32),
        )

        self.block4 = nn.Sequential(
            ##################### Dilated Convolution #####################
            # Layer 1
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=2,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(64),
            # Layer 2
            nn.Conv2d(
                in_channels=64,
                out_channels=96,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(96),
            # Layer 3
            nn.Conv2d(
                in_channels=96,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(64),
        )

        ##################### GAP #####################
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1))

        ##################### Fully Connected Layer #####################
        self.output_block = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 10))

    def print_view(self, x):
        """Print shape of the model"""
        if self.print_shape:
            print(x.shape)

    def forward(self, x):
        """Forward pass"""

        x = self.block1(x)
        self.print_view(x)
        x = self.block2(x)
        self.print_view(x)
        x = self.block3(x)
        self.print_view(x)
        x = self.block4(x)
        self.print_view(x)
        x = self.gap(x)
        self.print_view(x)
        # Flatten the layer
        x = x.view((x.shape[0], -1))
        self.print_view(x)
        x = self.output_block(x)
        self.print_view(x)
        x = F.log_softmax(x, dim=1)

        return x

