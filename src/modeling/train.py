import torch
import os
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
from src.dataset import get_mnist_dataloaders 

""" Training Module 

This module sets up a resnet model to required specifics 
Then using a loop, loads the training data in batches,
calculates loss, back propogates and then optomises weights 
saves the updated parameters   

/// Prerequisites ///

From torchvision.models import:
- 'resnet18' to build model

From torchvision import:
- transforms to preprosses image 

Import: 
- torch.nn to adjust layers in Residual Network model 

"""


def get_resnet18_mnist():
    
    ''' Get and set up ResNet model
    
    Function to load a resnet model and make adjustments to first 
    convolutional layer to reduce to 1 channel (grayscale) instead of RGB
    
    Returns: 
    adjusted model 
    
    '''
    
    model = resnet18(num_classes=10)  # 10 classes for MNIST
    # Change first conv layer to accept 1 channel (instead of 3 for RGB)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

def train_model(epochs=12, batch_size=128,device=None, model_path="models/resnet18_mnist.pth"):
    
    '''Training Function 
    
    Loops training data in batches of 128 images,
    applies transformations to each image,
    trains the model for specified number of epochs,
    calculates loss, back propogates and then optomises weights 
    saves the updated parameters   
    
    '''
    
    transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])
    # transforms input images from PIL format to PyTorch tensors.
    # uses .Compose from the transforms module to create a transformation pipeline
    
    # Set device
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    # assign device if GPU is available use it, if not use cpu
    
    train_loader, test_loader = get_mnist_dataloaders(batch_size)
    # get dataloaders with transformation applied 
    
    model = get_resnet18_mnist().to(device) 
    # initialise model and move to device 
    criterion = nn.CrossEntropyLoss()
    # set loss function 
    optimizer = torch.optim.Adam(model.parameters())
    # set optomiser, use adam 
    
    #start training loop 
    for epoch in range(epochs):
        
        # initialise model in training mode
        model.train()
        # initalise running_loss, correct, total all to 0
        running_loss, correct, total = 0.0, 0, 0
        
        # loop over images & labels from each batch
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        # loop over the training data
        for images, labels in loop:
            
            # move images and labels to device
            images, labels = images.to(device), labels.to(device)
            
            # reset gradients to zero before new backward pass
            optimizer.zero_grad()
            
            # peform a forward pass on current batch images
            outputs = model(images)
            
            # calculate loss between predicted and true labels
            loss = criterion(outputs, labels)
            
            # back propogate the loss - compute gradients for each parameter
            loss.backward()
            
            # update weights using Adam optimizer
            optimizer.step()
            
            # keep running total of accumlated loss, correct predictions, and total samples 
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=running_loss / (total / batch_size), acc=100. * correct / total)
            
            # print loss and accuracy
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss / len(train_loader):.4f} - Accuracy: {100 * correct / total:.2f}%")
       
    # save the model after 12 epochs
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
        