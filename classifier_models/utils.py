import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix
import yaml


class parse(object):
    """
    This class reads yaml parameter file and allows dictionary like access to the members.
    """
    def __init__(self, path):
        with open(path, 'r') as file:
            self.parameters = yaml.safe_load(file)

    # Allow dictionary like access
    def __getitem__(self, key):
        return self.parameters[key]

    def save(self, filename):
        with open(filename, 'w') as f:
            yaml.dump(self.parameters, f)


def train_classifier(classifier, epoch, optimizer, train_loader, device, logging, writer, n_steps=0):
    """helper for batch trainig of a pytorch model

    Args:
        classifier (obj): pytorch classification model object
        epoch (int): epoch number
        optimizer (obj): torch otim object representing optimizer algorithm
        train_loader (obj): torch DataLoader object representing train-set batches
        device (str): torch.device
        logging (obj): python logging instance
        writer (obj): tensorboard writer object
        n_steps (int, optional): number of spikes per incidence. Defaults to 0.

    Returns:
        float: total epoch loss normalized by total number of samples
    """
    total_loss_value = 0.0
    number_of_observations = len(train_loader.dataset)
    criterion = nn.BCELoss()

    for input_x, labels in train_loader:

        input_x = input_x.to(device, non_blocking=True)  # (N,C,H,W)
        labels = labels.to(device, non_blocking=True)
        # direct spike input
        if n_steps:
            input_x = input_x.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)  # (N,C,H,W,T)

        outputs = classifier(input_x)  # sampled_z(B,C,1,1,T)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss_value += loss.item()

    normalized_loss = total_loss_value / number_of_observations

    logging.info(f"Classification Train [{epoch}] Loss: {normalized_loss}")
    writer.add_scalar('Classification_Train/loss', normalized_loss, epoch)

    return normalized_loss


def test_classifier(classifier, epoch, test_loader, device, logging, writer, n_steps=0):
    """helper for batch testing of a pytorch model

    Args:
        classifier (obj): pytorch classification model object
        epoch (int): epoch number
        optimizer (obj): torch otim object representing optimizer algorithm
        test_loader (obj): torch DataLoader object representing test-set batches
        device (str): torch.device
        logging (obj): python logging instance
        writer (obj): tensorboard writer object
        n_steps (int, optional): number of spikes per incidence. Defaults to 0.

    Returns:
        float: total epoch loss normalized by total number of samples
    """
    total_loss_value = 0.0
    number_of_observations = len(test_loader.dataset)

    for input_x, labels in test_loader:
        input_x = input_x.to(device, non_blocking=True)  # (N,C,H,W)
        labels = labels.to(device, non_blocking=True)
        # direct spike input
        if n_steps:
            input_x = input_x.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)  # (N,C,H,W,T)

        outputs = classifier(input_x)  # sampled_z(B,C,1,1,T)

        criterion = nn.BCELoss()
        loss = criterion(outputs, labels)

        total_loss_value += loss.item()

    normalized_loss = total_loss_value/number_of_observations

    logging.info(f"Classification Test [{epoch}] Loss: {normalized_loss}")
    writer.add_scalar('Classification_Test/loss', normalized_loss, epoch)

    return normalized_loss


def return_model_accuracy(classifier, data_loader, device, n_steps=0):
    """calculates overall model accuracy for all samples in a data loader

    Args:
        classifier (obj): pytorch classification model object
        data_loader (obj): torch DataLoader object representing data-set batches
        device (str): torch.device
        n_steps (int, optional): number of spikes per incidence. Defaults to 0.
    Returns:
        float: overal accuracy percentage
    """
    correct = 0
    total = 0

    for input_x, labels in data_loader:
        input_x = input_x.to(device, non_blocking=True) # (N,C,H,W)
        labels = torch.argmax(labels, dim=2)
        labels = labels.to(device, non_blocking=True).to(torch.int64)
        if n_steps:
            input_x = input_x.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)  # (N,C,H,W,T)

        outputs = classifier(input_x)  # sampled_z(B,C,1,1,T)
        predicted = torch.argmax(outputs, dim=2)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    return 100 * correct / total


def return_model_metrics(classifier, data_loader, device, n_steps=0):
    """calculates overall classification metrics for all samples in a data loader

    Args:
        classifier (obj): pytorch classification model object
        data_loader (obj): torch DataLoader object representing data-set batches
        device (str): torch.device
        n_steps (int, optional): number of spikes per incidence. Defaults to 0.
    Returns:
        float: overal accuracy percentage
        float: overal specificity percentage
        float: overal sensitivity percentage
    """
    total_tn = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for input_x, labels in data_loader:
        input_x = input_x.to(device, non_blocking=True) # (N,C,H,W)
        labels = torch.argmax(labels, dim=2)
        labels = labels.to(device, non_blocking=True).to(torch.int64)
        if n_steps:
            input_x = input_x.unsqueeze(-1).repeat(1, 1, 1, 1, n_steps)  # (N,C,H,W,T)
        outputs = classifier(input_x)  # sampled_z(B,C,1,1,T)
        predicted = torch.argmax(outputs, dim=2)

        if (max(labels) == 0) and (max(predicted) == 0):
            total_tn += labels.size(0)
        elif (min(labels) == 1) and (min(predicted) == 1):
            total_tp += labels.size(0)
        else:
            tn, fp, fn, tp = confusion_matrix(labels.to('cpu'), predicted.to('cpu')).ravel()
            total_tn += tn
            total_fp += fp
            total_fn += fn
            total_tp += tp

    acc = 100 *  (total_tn + total_tp) / (total_tn + total_fp + total_fn + total_tp)
    specificity = 100 * total_tn / (total_tn + total_fp)
    sensitivity = 100 * total_tp / (total_tp + total_fn)

    return acc, specificity, sensitivity

