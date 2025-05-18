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
    """Makes train and test dataloaders

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
    # uses .Compose from the transforms module to create a transformation pipeline

    # Create datasets:

    train_dataset = datasets.MNIST(
        root="data/raw", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="data/raw", train=False, download=True, transform=transform
    )
    # load the two parts of the data set into the project directory data/raw
    # train=true/false parameter devides into test and train
    # download param = True means, if the data is not already in the directory it will be downloaded

    # Create dataloaders:

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # use the Dataloader module to wrap around the datasets, loading in batches of 128
    # shuffle for the training set to avoid overfitting
    # leave unshuffled for the testing set for reproduceability

    # return dataloaders:
    return train_loader, test_loader
