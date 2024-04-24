"""TCL"""





import os
import sys
import tensorflow as tf

import torch
import torch.nn as nn

# Equivalent of FLAGS in PyTorch using argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--FILTER_COLLECTION', default='filter', help='filter collection')
args = parser.parse_args()

# =============================================================
# =============================================================

def _variable_init(name, shape, wd, initializer = nn.init.xavier_uniform_, trainable = True):
    """Helper to create an initialized Variable with weight decay.

    Args:
        name: name of the variable
        shape: list of ints
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    var = torch.empty(shape)
    initializer(var)
    var = nn.Parameter(var, requires_grad=trainable)

    # Weight decay
    if wd is not None:
        weight_decay = wd * torch.sum(var ** 2) / 2

    return var, weight_decay
# =============================================================
# =============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class Maxout(nn.Module):
    def __init__(self, num_pieces):
        super(Maxout, self).__init__()
        self.num_pieces = num_pieces

    def forward(self, x):
        assert x.shape[1] % self.num_pieces == 0  # Number of in_features should be divisible by num_pieces
        return x.view(*x.shape[:1], x.shape[1] // self.num_pieces, self.num_pieces, *x.shape[2:]).max(dim=2)[0]

class MLP(nn.Module):
    def __init__(self, list_hidden_nodes, num_class, wd=1e-4, maxout_k=2, feature_nonlinearity='abs'):
        super(MLP, self).__init__()
        self.list_hidden_nodes = list_hidden_nodes
        self.num_class = num_class
        self.wd = wd
        self.maxout_k = maxout_k
        self.feature_nonlinearity = feature_nonlinearity
        self.layers = nn.ModuleList()

        for i in range(len(list_hidden_nodes)):
            if i != len(list_hidden_nodes) - 1:  # Not last layer
                self.layers.append(nn.Linear(list_hidden_nodes[i], list_hidden_nodes[i+1]*maxout_k))
                self.layers.append(Maxout(maxout_k))
            else:  # Last layer
                self.layers.append(nn.Linear(list_hidden_nodes[i], list_hidden_nodes[i+1]))

        self.final_layer = nn.Linear(list_hidden_nodes[-1], num_class)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1 and self.feature_nonlinearity == 'abs':
                x = torch.abs(x)
        feats = x
        logits = self.final_layer(x)
        return logits, feats
    
    
# =============================================================
# =============================================================
def loss(logits, labels):
    """Calculate cross entropy loss and accuracy.
    Args:
        logits: logits from the model.
        labels: labels from the data. 1-D tensor of shape [batch_size]
    Returns:
        Loss tensor of type float and accuracy.
    """
    # Calculate the average cross entropy loss across the batch.
    criterion = nn.CrossEntropyLoss()
    cross_entropy_loss = criterion(logits, labels)

    # Calculate accuracy
    _, predicted = torch.max(logits.data, 1)
    correct_prediction = (predicted == labels).sum().item()
    accuracy = correct_prediction / labels.size(0)

    return cross_entropy_loss, accuracy

# =============================================================
# =============================================================
from torch.utils.tensorboard import SummaryWriter

class MovingAverage():
    def __init__(self, factor=0.9):
        self.factor = factor
        self.raw_data = 0
        self.average = 0

    def update(self, raw_data):
        self.raw_data = raw_data
        self.average = self.factor * self.average + (1 - self.factor) * raw_data
        return self.average

def add_loss_summaries(total_loss, step, writer, losses, moving_averages):
    """Add summaries for losses in CIFAR-10 model.
        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.

    Args:
        total_loss: total loss from loss().
        step: current step in training.
        writer: TensorBoard SummaryWriter.
        losses: list of other losses.
        moving_averages: list of MovingAverage objects for each loss.
    """
    # Compute the moving average of all individual losses and the total loss.
    for i, loss in enumerate(losses + [total_loss]):
        avg = moving_averages[i].update(loss.item())

        # Log raw loss and moving average loss
        writer.add_scalar(losses[i].__class__.__name__ + ' (raw)', loss.item(), step)
        writer.add_scalar(losses[i].__class__.__name__, avg, step)

# =============================================================
# =============================================================
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import wandb

def train(model, total_loss, accuracy, initial_learning_rate, momentum, decay_steps, decay_factor, moving_average_decay=0.9999):
    """Train model.
        Create an optimizer and apply to all trainable variables. Add moving
        average for all trainable variables.
    Args:
        model: PyTorch model to train.
        total_loss: total loss from loss().
        accuracy: accuracy tensor
        initial_learning_rate: initial learning rate
        momentum: momentum parameter
        decay_steps: decay steps
        decay_factor: decay factor
        moving_average_decay: (option) moving average decay of variables to be saved
    Returns:
        None
    """
    # Create an optimizer
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=momentum)

    # Decay the learning rate exponentially based on the number of steps.
    scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_factor)

    # Compute gradients and update model parameters
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Update learning rate
    scheduler.step()

    # Compute moving averages of total loss and accuracy
    total_loss_avg = MovingAverage(moving_average_decay)
    accuracy_avg = MovingAverage(moving_average_decay)
    total_loss_avg.update(total_loss.item())
    accuracy_avg.update(accuracy.item())

    # Log metrics to wandb
    wandb.log({"total_loss": total_loss.item(), "accuracy": accuracy.item(),
               "total_loss_avg": total_loss_avg.average, "accuracy_avg": accuracy_avg.average,
               "learning_rate": scheduler.get_last_lr()[0]})