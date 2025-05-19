import pytest
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

'''Tests for dataset.py

UNIT TESTS 
To check individual components and parameters:
- transforms and .Compose < 
- loading datasets 
- Dataloader 

FUNCTION TESTS 
- get_mnist_dataloaders


'''

def test_transform_pipeline():
    ''' Transform pipeline test 
    
    Test if the transform pipeline works as expected 
    first create a dummy PIL imageof correct dimensions
    then transform it to a Tensor and check tensor is correct data type and shape
    
    '''
    
    from PIL import Image
    import numpy as np
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    dummy_img = Image.fromarray((np.random.rand(28,28) * 255).astype('uint8'))
    # create an array of 28 x 28 with random numbers, multiply 255 to get equivilent of grayscale range
    # image.fromarray turns array into a PIL image 
    # type uint8 is an unsigned 8bit int, standard for for image data  
    
    tensor_img = transform(dummy_img)
    
    assert tensor_img.dtype == torch.float32
    assert tensor_img.shape == (1, 28, 28)
    
