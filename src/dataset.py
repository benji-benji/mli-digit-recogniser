from torchvision import datasets, transforms
from torch.utils.data import DataLoader


""" Dataset Module 

This module downloads and prepares the MNIST dataset,
It defines a function which produces two dataloaders:
one for the training set and one for the test set. 

/// Prerequisites ///

From torchvision (part of pytorch) import:
- 'datasets' for accessing datasets
- 'transforms' for applying image transformations  

From torch.utils.data import:
- 'dataloader' for batching and loading data 

"""


def get_mnist_dataloaders(batch_size=128):
    """ Makes train and test dataloaders
    
    This function downloads MNIST dataset,
    converts images from MNIST dataset into Pytorch tensors,
    loads the training and testing parts ( 60k and 10k respectively)
    creates the two dataloaders and returns them.
    
    Parameters
    ----------
    Batch_size is set to 128, because otherwise dataloaders default to 1
    
    Returns
    ----------
    (Training Dataloader, Testing Dataloader)  
    
    """
    transform = transforms.Compose([transforms.ToTensor()])
    # transforms input images from PIL format to PyTorch tensors.
