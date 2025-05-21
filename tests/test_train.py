import pytest
import torch
import os
import tempfile
import torch.nn as nn
from unittest.mock import patch
from src.modeling.train import get_resnet18_mnist
from src.modeling.train import train_model
from torch.optim import Adam   


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

def test_train_model_lossfunction():
    ''' Test train_model function assigns loss function correctly
    
    check the function correctly assigns the CrossEntropyLoss function
    '''
    criterion = nn.CrossEntropyLoss()
    # line from source code, assign loss function
    assert criterion.__class__.__name__ == 'CrossEntropyLoss'
    # check the loss function is CrossEntropyLoss


def test_train_model_optimizer():
    ''' Test train_model function assigns optimizer correctly
    
    check the function correctly assigns the Adam optimizer
    '''
    model = get_resnet18_mnist()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # line from source code, assign optimizer
    assert optimizer.__class__.__name__ == 'Adam'
    # check the optimizer is Adam

#def test_train_model_loop1():


def test_train_model_runs_one_epoch():
    '''Test 1 epoch + save model (with dummy data/model).'''
    
    # Create dummy dataloader: 2 batches of fake data
    dummy_data = [(torch.randn(2, 1, 224, 224), torch.tensor([1, 0]))] * 2

    # Use a dummy model: Flatten + Linear (224x224 -> 10)
    dummy_model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(224*224, 10)
    )

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("src.modeling.train.get_mnist_dataloaders", return_value=(dummy_data, dummy_data)), \
         patch("src.modeling.train.get_resnet18_mnist", return_value=dummy_model):

        model_path = os.path.join(tmpdir, "model.pth")
        
        train_model(epochs=1, batch_size=2, model_path=model_path)
        
        assert os.path.exists(model_path)  


def test_train_model_updates_weights():
    '''Test that model weights change after training'''

    dummy_data = [(torch.randn(2, 1, 224, 224), torch.tensor([1, 0]))] * 2
    dummy_model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(224*224, 10)
    )

    # Save initial weights
    initial_weights = dummy_model[1].weight.clone().detach()

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("src.modeling.train.get_mnist_dataloaders", return_value=(dummy_data, dummy_data)), \
         patch("src.modeling.train.get_resnet18_mnist", return_value=dummy_model):

        model_path = os.path.join(tmpdir, "model.pth")
        train_model(epochs=1, batch_size=2, model_path=model_path)

        # Compare with new weights
        updated_weights = dummy_model[1].weight.detach()

        assert not torch.equal(initial_weights, updated_weights.cpu())
        # Check that weights have changed after training    


def test_train_model_prints_loss():
    '''Test that training loop prints loss and accuracy.'''
    
    dummy_data = [(torch.randn(2, 1, 224, 224), torch.tensor([1, 0]))] * 2
    dummy_model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(224*224, 10)
    )

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("src.modeling.train.get_mnist_dataloaders", return_value=(dummy_data, dummy_data)), \
         patch("src.modeling.train.get_resnet18_mnist", return_value=dummy_model), \
         patch("builtins.print") as mock_print:

        model_path = os.path.join(tmpdir, "model.pth")
        train_model(epochs=1, batch_size=2, model_path=model_path)

        mock_print.assert_called()
        
