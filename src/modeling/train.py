import torch
import os
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

def train_model(epochs=12, batch_size=128, model_path="models/resnet18_mnist.pth"):
    
    '''Training Function 
    
    Loops training data in batches of 128 images,
    calculates loss, back propogates and then optomises weights 
    saves the updated parameters   
    
    '''
    
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize MNIST images for ResNet
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # Standard MNIST normalization
    ])
    # transforms input images from PIL format to PyTorch tensors.
    # uses .Compose from the transforms module to create a transformation pipeline
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        ''' Training loop
        
        initalise model and set key variables to 0
        for each epoch,
        for each batch in the training data
        - get the images and labels
        - move to device
        - zero the gradients
        - forward pass
        - calculate loss
        - backward pass
        - update weights
        - calculate running loss
        - calculate accuracy
        - print the loss and accuracy every 128 batches
        - save the model after 12 epochs

        '''
        
        # initialise model in training mode
        model.train()
        # initalise running_loss, correct, total all to 0
        running_loss, correct, total = 0.0, 0, 0
        
        # loop over images & labels from each batch
        
        for images, labels in train_loader:
            
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
            
            # print loss and accuracy
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss / len(train_loader):.4f} - Accuracy: {100 * correct / total:.2f}%")
       
    # save the model after 12 epochs
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
        