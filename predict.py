# Imports here
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
import data_load_utility # Import utility functions for loading data and making predictions
import model_func # Import utility functions for model-related operations


# Parse command-line arguments
parser = argparse.ArgumentParser(description = 'This is the parser for predict.py')

parser.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type = str)
parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

args = parser.parse_args()
image_path = args.input
outputs_number = args.top_k
device = args.gpu
json_name = args.category_names
path = args.checkpoint

# Define the main function
def main():
    # Load the trained model
    model=model_func.load_checkpoint(path)
    
    # Load the category-to-name mapping from a JSON file
    with open(json_name, 'r') as json_file:
        name = json.load(json_file)
        
    # Make predictions using the model
    ps = model_func.predict(image_path, model, outputs_number)
    a = np.array(ps[0][0].cpu())
    b = [name[str(index + 1)] for index in np.array(ps[1][0].cpu())]
    
    i = 0
    while i < outputs_number:
        print("{} with a probability of {}".format(b[i], a[i]))
        i += 1
    print("Finished Predicting!")

    
if __name__== "__main__":
    main()