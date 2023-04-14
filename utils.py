# third party
import numpy as np
import torch
import tqdm
from sklearn import metrics


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


def build_patient_lateral_predictions_and_evaluate(y_pred, data):
    """
    Function that takes in a dataframe with patient information
    and the output will be the predictions for a patients L & R
    breast
    """
    # Get the predictions
    data["predictions"] = y_pred

    # Get predictions df
    predictions_df = (
        data.groupby(["patient_id", "laterality"])[["cancer", "predictions"]]
        .mean()
        .reset_index(drop=True)
    )

    # Get the different metrics
    y_true = predictions_df["cancer"].values
    y_pred = predictions_df["predictions"].values

    model_metrics = {}
    # Log loss
    model_metrics["logloss"] = metrics.log_loss(y_true, y_pred)

    # AUCROC score
    model_metrics["roc_auc_score"] = metrics.roc_auc_score(y_true, y_pred)

    # Macro F1 score
    f1_scores = []
    for i in np.arange(0.05, 0.50, 0.01):
        binary_predictions = (y_pred > i).astype(int)
        f1_score = pfbeta(y_true, binary_predictions, beta=1)
        f1_scores.append((f1_score, i))

    f1_scores = sorted(f1_scores)
    f1_score = f1_scores[0]
    model_metrics["f1_score"] = f1_score

    return model_metrics


def pfbeta(labels, predictions, beta):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if labels[idx]:
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
        return result
    else:
        return 0
