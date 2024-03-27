"""Module to define utility functions for the project."""
import torch


def get_device():
    """
    Function to get the device to be used for training and testing.
    """

    # Check if cuda is available
    cuda = torch.cuda.is_available()

    # Based on check enable cuda if present, if not available
    if cuda:
        final_choice = "cuda"
    else:
        final_choice = "cpu"

    # pylint: disable=E1101
    return final_choice, torch.device(final_choice)


def get_correct_prediction_count(prediction, label):
    """
    Function to get the count of correct predictions.
    """
    return prediction.argmax(dim=1).eq(label).sum().item()


# Function to save the model
def save_model(epoch, model, optimizer, scheduler, batch_size, criterion, file_name):
    """
    Function to save the trained model along with other information to disk.
    """
    # print(f"Saving model from epoch {epoch}...")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "batch_size": batch_size,
            "loss": criterion,
        },
        file_name,
    )
