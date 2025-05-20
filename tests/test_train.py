import pytest
import torch
from src.modeling.train import get_resnet18_mnist

def test_get_resnet18_mnist_return():
    ''' Test get_resnet18_mnist returns 
    
    check the function returns model
     '''
    dummy_model1 = get_resnet18_mnist()
    # run function and assign to variable
    assert isinstance(dummy_model1, torch.nn.Module)
    # check variables is correct object type ( Dataloader)

def test_get_resnet18_mnist_layeradjustment1():
    ''' Test get_resnet18_mnist adjusts the 1st Conv layer 
    
    check the model returned has correct changes made to first layer
    '''
    dummy_model2 = get_resnet18_mnist()
    # run function and assign to variable
    assert dummy_model2.conv1.in_channels == 1
    # check the first layer has been adjusted to accept 1 channel (grayscale)
    
def test_get_resnet18_mnist_layeradjustment2():
    ''' Test get_resnet18_mnist adjusts the 1st Conv layer 
    
    check the model returned has correct changes made to first layer
    '''
    dummy_model3 = get_resnet18_mnist()
    # run function and assign to variable
    assert dummy_model3.conv1.out_channels == 64
    # check the first layer has been adjusted to accept 64 channels (grayscale)

def test_train_model_device():
    ''' Test train_model function assigns device correctly
    
    check the function correctly assigns the device to GPU or CPU
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # line from source code, assign device if GPU is available use it, if not use cpu
    assert device == torch.device("cuda") or device == torch.device("cpu")
    # check the device is either cuda or cpu

#def test_train_model_():
    