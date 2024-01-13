import numpy as np

from tqdm.notebook import tqdm

import torch

import torch.nn as nn # Defines a series of classes to implement neural nets
import torch.nn.functional as F # Contains functions that are used in network layers

from torch.utils.data import DataLoader

from torchmetrics import Accuracy


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')



################
# FUNCTIONS
################

def train(model: nn.Module,
          num_epochs: int,
          device: str,
          loss_module,
          optimizer,
          train_loader:DataLoader,
          valid_loader:DataLoader = None):
    
    """ Function to train a Neural Network """
    
    total_steps = len(train_loader)
    
    history = []
    
    for epoch in tqdm(range(num_epochs)):
        
        train_loss = 0.0
        valid_loss = 0.0
        
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            
            model.train()
            
            images = images.to(device)
            labels = labels.to(device)
            
            preds = model(images)
            loss = loss_module(preds, labels)
            
            train_loss += loss.item()
            
            #Backward and Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
        train_loss /= len(train_loader)
        
        if valid_loader:
            
            # Validation
            with torch.no_grad():

                for images, labels in valid_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    preds = model(images)

                    loss = loss_module(preds, labels)
                    valid_loss += loss.item()

                    valid_loss /= len(valid_loader)
                    del images, labels, outputs

        
            
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss}, Validation Loss: {valid_loss}")
                
        history.append([train_loss, valid_loss])
        
    sns.lineplot([h[0] for h in history], color='navy', label='Train Loss').set_title('Loss vs Epoch')
    sns.lineplot([h[1] for h in history], color='orange', label='Validation Loss').set_title('Loss vs Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
        
    return history




def eval_model(model:nn.Module,
               num_classes:int,
               device:str,
               dataloader:DataLoader,
               loss_module):
    
    """Function to computer the accuracy of a neural network"""
    
    test_loss, test_acc = 0, 0

    model.to(device)

    accuracy = Accuracy(task='multiclass', num_classes=num_classes)
    accuracy.to(device)

    model.eval()

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            loss += loss_module(y_pred, y)
            acc += accuracy(y_pred, y)

        loss /= len(dataloader)
        acc /= len(dataloader)

    print(f"Loss: {loss: .5f}|  Accuracy: {acc: .5f}")