# Contains utility functions like loading data and preprocessing images
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



def load_data(root = "./flowers"):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    data_dir = root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])
    valid_and_test_transform = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])
    

    # TODO: Load the datasets with ImageFolder
    image_datasets = ImageFolder(root=train_dir, transform=data_transforms)
    valid_data = ImageFolder(root=valid_dir, transform=valid_and_test_transform)
    test_data = ImageFolder(root=test_dir, transform=valid_and_test_transform)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloaders = data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    valid_data_loader = data.DataLoader(valid_data, batch_size=64)
    test_data_loader = data.DataLoader(test_data, batch_size=64)

    return trainloaders, valid_data_loader, test_data_loader , image_datasets