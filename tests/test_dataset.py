import pytest
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.dataset import get_mnist_dataloaders 



'''Tests for dataset.py

UNIT TESTS 
To check individual components and parameters:
- transforms and .Compose < 
- loading datasets 
- Dataloader 

FUNCTION TESTS 
- get_mnist_dataloaders


'''

#UNIT TESTS 

def test_transform_pipeline():
    '''Test Transform pipeline
    
    Test if the transform pipeline works as expected 
    first create a dummy PIL imageof correct dimensions
    then transform it to a Tensor and check tensor is correct data type and shape
    
    '''
    
    from PIL import Image
    import numpy as np
    
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize MNIST images for ResNet
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # Standard MNIST normalization
    ])
    # line of code from dataset.py to test 
    
    dummy_img = Image.fromarray((np.random.rand(28,28) * 255).astype('uint8'))
    # create an array of 28 x 28 with random numbers, multiply 255 to get equivilent of grayscale range
    # image.fromarray turns array into a PIL image 
    # type uint8 is an unsigned 8bit int, standard for for image data  
    
    tensor_img = transform(dummy_img)
    # run transformation 
    
    assert tensor_img.dtype == torch.float32
    # check data type is float32 
    assert tensor_img.shape == (1, 224, 224)
    # check the tensor is correct shape


def test_train_dataset():
    ''' Test MNIST training dataset
    
    Check datasets.MNIST loads correctly
    size / shape of dataset:  
    check first instance x is a tensor
    check first instance y is a int 
    '''
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize MNIST images for ResNet
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # Standard MNIST normalization
    ])
    # line of code from dataset.py to test 
    
    train_dataset = datasets.MNIST(root="data/raw", train=True, download=True, transform=transform)
    # line of code from dataset.py to test 
    
    assert len(train_dataset) == 60000
    # check  dataset contains 60000 entries  
    x, y = train_dataset[0]
    # check two variables of first instance 
    assert isinstance(x, torch.Tensor)
    # check data type of x variable (tensor representing handwritten digit)
    assert x.shape == (1, 224, 224)
    # check shape of x variable 
    assert isinstance(y, int)
    # check y is integer ( correct digit)
    
def test_test_dataset():
    ''' Test MNIST test dataset
    
    Check datasets.MNIST loads correctly
    size / shape of dataset:  
    check first instance x is a tensor
    check first instance y is a int 
    '''
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize MNIST images for ResNet
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # Standard MNIST normalization
    ])
    
    test_dataset = datasets.MNIST(root="data/raw", train=False, download=True, transform=transform)
    
    assert len(test_dataset) == 10000
    # check  dataset contains 60000 entries  
    x, y = test_dataset[0]
    # check two variables of first instance 
    assert isinstance(x, torch.Tensor)
    # check data type of x variable (tensor representing handwritten digit)
    assert x.shape == (1, 224, 224)
    # check shape of x variable 
    assert isinstance(y, int)
    # check y is integer ( correct digit)

def test_dataloader_batching():
    '''Test batching
    
    Using train_dataset
    initialise a test dataloader
    iterate the loader 
    check the size and shape of batches is correct 
    
    '''
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize MNIST images for ResNet
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # Standard MNIST normalization
    ])
    
    train_dataset = datasets.MNIST(root="data/raw", train=True, download=True, transform=transform)
    # lines of code from source dataset.py
    test_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    # create dummy dataloader 'loader'
    test_loader_images, test_loader_labels = next(iter(test_loader))
    # interate over the dummy dataloader, and assign the items in tuple into 2 variables 
    # first for the batch of images
    # second for the batch of corresponding lables 
    assert test_loader_images.shape == (128, 1, 224, 224)
    # check shape of the image variable 
    assert test_loader_labels.shape == (128,)
    # check shape of the labels variable 


# FUNCTION TESTS 

# tests for get_mnist_dataloaders function 

def test_get_mnist_dataloaders_1():
    ''' Test get_mnist_dataloaders returns 
    
    check the function returns dataloaders 
     '''
    train_loader, test_loader = get_mnist_dataloaders(batch_size=128)
    # run function and assign the items in tuple to variables 
    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    # check variables are correct object types ( Dataloader)

def test_get_mnist_dataloaders_2():
    '''Check if images and labels from dataloader have expected shapes
    
    iterate and check the function returns correctly shaped batches 
    '''
    
    train_loader, _ = get_mnist_dataloaders(batch_size=128)
    
    x, y = next(iter(train_loader))
    
    assert x.shape == (128, 1, 224, 224)
    assert y.shape == (128,)
    

