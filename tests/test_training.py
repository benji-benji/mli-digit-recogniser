import pytest
import torch
from models.training import get_resnet18_mnist

def test_get_resnet18_mnist_1():
    ''' Test get_resnet18_mnist returns 
    
    check the function returns model
     '''
    dummy_model = get_resnet18_mnist()
    # run function and assign the items in tuple to variables 
    assert isinstance(dummy_model, torch.nn.Module)
    # check variables are correct object types ( Dataloader)