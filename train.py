import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils import data
from PIL import Image
import numpy as np
import os, random
import json
import signal
import torchvision

from contextlib import contextmanager
from collections import OrderedDict 

import requests


import matplotlib.pyplot as plt
from matplotlib.pyplot import FormatStrFormatter

from torch.autograd import Variable
import data_load_utility
import model_func


# Parse command-line arguments
parser = argparse.ArgumentParser(
    description = 'This is the Parser for train.py'
)
parser.add_argument('data_dir', action="store", default="./flowers/")
parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
parser.add_argument('--arch', action="store", default="vgg16")
parser.add_argument('--learning_rate', action="store", type=float,default=0.001)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=25088)
parser.add_argument('--epochs', action="store", default=3, type=int)
parser.add_argument('--gpu', action="store", default="gpu")

args = parser.parse_args()
data_path = args.data_dir
path = args.save_dir
lr = args.learning_rate
struct = args.arch
hidden_units = args.hidden_units
power = args.gpu
epochs = args.epochs

# Check if GPU is available
if torch.cuda.is_available() and power == 'gpu':
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
# Define the main function
def main():
    # Load data
    trainloaders, valid_data_loader, test_data_loader , image_datasets = data_load_utility.load_data(data_path)
    # Setup the neural network and criterion
    model, criterion, optimizer = model_func.nn_setup(struct,lr, hidden_units)
  
    
    # Train Model
    print_every = 5
    steps = 0
    loss_show = []

    for e in range(epochs):
        running_loss = 0
        for inputs, labels in trainloaders:
            steps += 1
        
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
            optimizer.zero_grad()
        
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in valid_data_loader:
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    
                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()
                    
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {e+1}/{epochs}.. "
                    f"Loss: {running_loss/print_every:.3f}.. "
                    f"Validation Loss: {valid_loss/len(valid_data_loader):.3f}.. "
                    f"Accuracy: {accuracy/len(valid_data_loader):.3f}")
                running_loss = 0
                model.train()
    # Save the checkpoint
    model_func.save_checkpoint(image_datasets,model,path,struct, hidden_units,lr,1)
    print("Saved checkpoint!")
# Execute the main function if this script is run
if __name__ == "__main__":
    main()