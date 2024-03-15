import random
import torch
import numpy as np
from datetime import datetime
import socket
import os


def unique_identifier():
    return datetime.now().strftime("%b%d_%H-%M-%S") + '_' + socket.gethostname()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, verbose=True, delta=1e-2, save_path=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.loss_min = np.Inf
        self.delta = delta
        if save_path is None:
            self.save_path = os.path.join('runs', datetime.now().strftime("%b%d_%H-%M-%S") + '_' +
                                          socket.gethostname(), 'checkpoint.pt')
        else:
            self.save_path = os.path.join(save_path, 'checkpoint.pt')
        dir = os.path.dirname(self.save_path)
        if not os.path.exists(dir):
            os.makedirs(dir)

    def __call__(self, loss, model):

        if self.loss_min > loss:
            self.save_checkpoint(loss, model)
            if self.loss_min / loss - 1 > self.delta:
                self.counter = 0
            else:
                self.counter += 1
            self.loss_min = loss
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            

    def save_checkpoint(self, loss, model):
        '''Saves model when training loss decrease.'''
        if self.verbose:
            print(
                f'Training loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model, self.save_path)
        self.loss = loss

def _fidelity(y1, y2):
    N = y1.shape[0] // 2
    r1 = y1[:N]
    i1 = y1[N:]
    r2 = y2[:N]
    i2 = y2[N:]

    R = torch.sum(r1*r2 + i1*i2)
    I = torch.sum(r1*i2-r2*i1)

    return (R**2 + I**2) / (torch.dot(y1, y1) * torch.dot(y2, y2))


fidelity = torch.vmap(_fidelity)

def fidelity_loss(y1, y2):
    return 1 - torch.mean(fidelity(y1, y2))


def train(model, loader, loss_fn, optimizer, device):
    model.train()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()


def test(model, loader, loss_fn, device, epoch=None, writer=None, verbose=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
    average_loss = total_loss / len(loader)

    if verbose:
        print(f"Training loss: {average_loss:.8f}")

    if writer is not None:
        # Log the average validation loss to TensorBoard
        writer.add_scalar('Training/Loss', total_loss / len(loader), epoch)

    return average_loss
