import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os.path
import time
import numpy as np
from tcl_pytorch.model import TCL,TCL_new
import torch.utils.data as data

import logging

logger = logging.getLogger(__name__)
def train(dataset,
          num_class,
          list_hidden_nodes,
          initial_learning_rate,
          momentum,
          max_steps,
          decay_steps,
          decay_factor,
          batch_size,
          train_dir,
          moving_average_decay=0.9999,
          summary_steps=500,
          checkpoint_steps=10000,
          MLP_trainable=True,
          save_file='model.pth',
          load_file=None,
          random_seed=None):
    
    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)



    # Define your model
    model = TCL(input_size=dataset.__getinputsize__(), list_hidden_nodes=list_hidden_nodes, num_class=num_class,MLP_trainable=MLP_trainable)

    if load_file:
        load_path = os.path.join(train_dir, load_file)
        if os.path.exists(load_path):
            state_dict = torch.load(load_path)
            model.load_state_dict(state_dict)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=momentum)

    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_factor)

    train_data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
   
    # Start training
    
    for step in range(max_steps):
        i=0
        accuracy=0
        losses=[]
        for data_inputs, data_labels in train_data_loader:
            start_time = time.time()

            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            x_batch = data_inputs
            y_batch = data_labels

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            logits, _ = model(x_batch)

            # Compute the loss
            loss = criterion(logits, y_batch)
            losses.append(loss.item())
            i+=1

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            duration = time.time() - start_time

            accuracy += calculate_accuracy(logits, y_batch)

        if step % 100 == 0:
            num_examples_per_step = batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

        logger.info('%s: step %d, lr = %f, loss = %.2f, accuracy = %.2f (%.1f examples/sec; %.3f sec/batch)' %
                    (datetime.now(), step, optimizer.param_groups[0]['lr'], np.mean(losses), accuracy/i * 100,
                    examples_per_sec, sec_per_batch))

            # if step % summary_steps == 0:
            #     # Add summary

            # if step % checkpoint_steps == 0:
            #     # Save checkpoint

        # scheduler.step()

    # Save trained model
    save_path = os.path.join(train_dir, save_file)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Save model in file: {save_path}")



def calculate_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy
