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
from vgg import vgg
from resnet import resnet



def test(model, testloader,criterion):
    '''
    This function takes two arguments and returns None
    
    Parameters:
        -model: Trained Image Classification Network
        -test_loader: DataLoader for test dataset
        
    Returns:
        None
    '''
    model.eval()
    correct = 0
    total = 0
    loss_total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    
    return loss_total,100 * correct / total


def train(model, train_loader, criterion , optimizer, epoch):
    '''
    This function takes five arguments and returns None
    
    Parameters:
        -model: Untrained Image Classification Network
        -train_loader: DataLoader for train dataset
        -criterion: Loss Function
        -optimizer: The optimization algorithm to use
        -epoch: Epoch Number
        
    Returns:
        None
    '''
    total_step = len(train_loader)
    train_loss = 0
    for i, (inputs,labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()


    train_loss /= len(train_loader)
    return train_loss


def main(args):
    

    
    if args.arch == "vgg":
        model = vgg(args.hidden_units)
        model = model.to(device)
    
    elif args.arch == "resnet":
        model = resnet(args.hidden_units)
        model = model.to(device)
        
    train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(size=224,scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
            ])

    test_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225])
                ])

    # Load the datasets with ImageFolder
    image_dataset_train = torchvision.datasets.ImageFolder(root = args.data_dir+"/train",transform = train_transform)
    image_dataset_test = torchvision.datasets.ImageFolder(root = args.data_dir+"/test",transform = test_transform)
    image_dataset_val = torchvision.datasets.ImageFolder(root = args.data_dir+"/valid",transform = test_transform)

    
    # Using the image datasets and the trainforms, define the dataloaders
    BATCH_SIZE = 64
    train_dataloader = DataLoader(image_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(image_dataset_val, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(image_dataset_test, batch_size=BATCH_SIZE, shuffle=True)    
    

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(),lr=args.learning_rate)
    
    print("Training Initiated")
    for epoch in range(args.epochs):
        
        train_loss = train(model, train_dataloader, criterion ,optimizer,epoch)
        val_loss, val_acc = test(model, val_dataloader, criterion)

        print("Epoch: {}/{}".format(epoch+1, args.epochs))
        print("Training Loss: {:.4f}".format(train_loss))
        print("Validation Loss: {:.4f} Validation Accuracy: {:.2f}%".format(val_loss, val_acc))
        torch.save({
            'epoch': epoch,
            'model': model,
            'optimizer_state_dict': optimizer.state_dict(),
            'class_to_idx' : image_dataset_train.class_to_idx
            }, args.save_dir + "model.pt")
    





if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Image Classification Project 2")
    parser.add_argument("data_dir",
                        type=str,
                        default="flowers",
                        metavar="Data_directory_path",
                        help="input data_directory for training (default: ./flowers)",
                       )

    parser.add_argument("--save_dir",
                        type=str,
                        default='./',
                        metavar="Model_check_point_save_dir",
                        help="Path for trained Model"
                       )
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.01,
                        metavar="LR",
                        help="learning rate (default: 0.01)",
                       )

    parser.add_argument("--hidden_units",
                        type=float,
                        default=512,
                        metavar="Num Neurons",
                        help="Hidden layer neurons",
                       )

    parser.add_argument("--epochs",
                        type=int,
                        default=2,
                        metavar="N",
                        help="Num_epochs"
                       )

    parser.add_argument("--gpu", 
                        type=bool, 
                        default=True,
                        help="Training on GPU")

    parser.add_argument("--arch", 
                        type=str, 
                        default="vgg",
                        help="Architecture for training")

    args = parser.parse_args()


    if args.gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    main(args)
