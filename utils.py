# third party
import numpy as np
import torch
import tqdm


# Train one epoch
def train_one_epoch(model, data_loader, epoch, criterion, optimizer, scaler, device):
    """
    Function to train one epoch of our image classifier
    """
    # Put model in training model
    model.train()

    # Training loss
    all_predictions = []
    all_labels = []
    train_loss = 0

    for index, (images, labels, weights) in tqdm.tqdm(enumerate(data_loader)):
        # Make sure the images are compatible with the model
        # and located on the right device
        images = images.to(device=device)
        labels = labels.to(device=device)

        # Zero out the gradients
        optimizer.zero_grad()

        # Used mixed precision for modeling
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            # Get the predictions
            predictions = model(images)
            sigmoid_predictions = torch.sigmoid(predictions)

            # Calculate the loss
            loss = criterion(predictions, labels)

        # Take the backward step
        scaler.scale(loss).backward()

        # Step through the optimizer
        scaler.step(optimizer)
        scaler.update()

        # accumulate the loss
        train_loss += loss.item()

        # Accumlate labels as well
        all_predictions.append(sigmoid_predictions.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    print(f"Epoch = {epoch} / loss = {train_loss / len(data_loader)}")
    all_predictions = np.concatenate(all_predictions).flatten().astype(np.float64)
    all_labels = np.concatenate(all_labels).flatten().astype(np.float64)
    return all_predictions, all_labels


def evaluate_model(model, data_loader, metric, device):
    """
    Function to evaluate the model. The metrics can be a dictionary
    that will run multiple metrics for the model
    """
    model.eval()

    # Make sure we turn off the ability to change / update gradients
    with torch.no_grad():
        predictions, labels = build_predictions(model, data_loader, device)

        # Assume sklearn metrics
        metric_value = metric(labels, predictions)
        print(f"Evaluation metric = {metric_value}")


def build_predictions(model, data_loader, device):
    """
    Function to simply build predictions for the new model
    """
    all_predictions = []
    all_labels = []
    model.eval()

    # Make sure we turn off the ability to change / update gradients
    # TODO: will have to do something if the labels are empty
    # I think this will be present in the data loader
    with torch.no_grad():
        for index, (images, labels, weights) in tqdm.tqdm(enumerate(data_loader)):
            images = images.to(device=device)

            # Get the predictions
            predictions = model(images)

            # With classification do not forget that we need
            # to add a sigmoid layer afterwards for probablistic
            # predictions
            predictions = torch.sigmoid(predictions)

            all_predictions.append(predictions.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    return np.concatenate(all_predictions), np.concatenate(all_labels)
