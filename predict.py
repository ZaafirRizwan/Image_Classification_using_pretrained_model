import numpy as np 
import torchvision
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import argparse
import json

def load_checkpoint(path):
    '''
        Load Model Checkpoint
    
    '''

    checkpoint = torch.load(path)
    model = checkpoint['model']
    class_to_idx = checkpoint['class_to_idx']

    model.eval()
    return model,class_to_idx


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
    ])
    
    
    im = Image.open(image)
    im = test_transform(im)
        
    return im


def predict(image_path, model,class_idx,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = torch.unsqueeze(image,0).to(device)
    result = model(image)
    result = nn.functional.softmax(result,dim=1)

    prob = torch.topk(result, topk)[0][0].tolist()
    indices = torch.topk(result, topk)[1][0].tolist()
    
    for i in range(len(indices)):
        for key, val in class_idx.items():
            if indices[i] == val:
                indices[i] = key
    
    return prob,indices


def main(args):
    
    
    model,class_idx = load_checkpoint(args.checkpoint)
    max_val,max_idx = predict(args.image_Path, model,class_idx,args.topk)
    

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    
    
    name = []
    for i in range(len(max_idx)):
        for key, val in cat_to_name.items():
            if max_idx[i] == key:
                name.append(val)
                break

                
    for i in range(len(max_val)):
        print("Probability: {prob}, Class_Name:{catname}".format(prob=max_val[i],catname=name[i]))





if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Deep Learning on Amazon Sagemaker")
    
    parser.add_argument("image_Path",
                       type=str,
                       default="flower/test/10/image_07090.jpg",
                       metavar="img_path",
                       help="Image Path on which Inference to be performed"
                       )
    
    
    parser.add_argument("checkpoint",
                        type=str,
                        default="./model.pt",
                        metavar="Checkpoint",
                        help="Model Checkpoint Path",
                       )
    

    parser.add_argument("--topk",
                   type=int,
                   default=3,
                   metavar="topk",
                   help="top k predictions",
                   )
    
    parser.add_argument("--category_names",
                   type=str,
                   default="cat_to_name.json",
                   metavar="cat_to_name",
                   help="Name of classes",
                   )
    
    parser.add_argument("--gpu", 
                        type=bool, 
                        default=True,
                        help="Training on GPU")
    
    args = parser.parse_args()
    
    if args.gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    
    
    main(args)