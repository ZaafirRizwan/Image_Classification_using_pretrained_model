import numpy as np 
import torchvision
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image


def vgg(num_neurons):
    '''
    This function takes one parameters and returns a Network
    
    Parameters:
        num_neurons: Num_neurons
        
    Returns:
        Untrained Image Classification Model
        
    '''
    pretrained_model = models.vgg16(pretrained=True)
    
    # Freezing Pretrained Weights
    pretrained_model.features.require_grad = False
    
    # Append Fully_Connected layer
    num_ftrs = pretrained_model.classifier[0].in_features
    
    pretrained_model.classifier = nn.Sequential( nn.Dropout(0.2),
                                                 nn.Linear(num_ftrs,num_neurons),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.2),
                                                 nn.Linear(num_neurons,num_neurons),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.2),
                                                 nn.Linear(num_neurons,102),
                                                
                                               )
        
    pretrained_model.classifier.require_grad = True

#     pretrained_model = pretrained_model.to(device)
    
    return pretrained_model

    