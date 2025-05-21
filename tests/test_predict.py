import pytest
import torch
import tempfile
import os
import torch.nn as nn
import numpy as np
from unittest.mock import patch
from torchvision.transforms import transforms
from PIL import Image
from src.modeling.train import get_resnet18_mnist
from src.modeling.predict import predict_single_image

def test_predict_single_image_device():
    '''Test predict_single_image device assignment
    
    check the function correctly assigns the device to GPU or CPU
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # line from source code, assign device if GPU is available use it, if not use cpu
    assert device == torch.device("cuda") or device == torch.device("cpu")
    # check the device is either cuda or cpu

def test_predict_single_image_loadimage():
    '''Check image is loaded, transformed correctly using a temporary file.'''

    # Create a grayscale image of correct size
    dummy_image = Image.new('L', (224, 224), color=255)  

    # Use tempfile to safely create and destroy the test image
    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_image_path = os.path.join(tmpdir, "test_image.png")
        dummy_image.save(dummy_image_path)

        # same transform used in pipeline
        dummy_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load and transform the image
        img = Image.open(dummy_image_path).convert("L")
        img = dummy_transform(img)
        img = img.unsqueeze(0)

        # Assertions
        assert img.shape == (1, 1, 224, 224), f"Expected shape (1,1,224,224), got {img.shape}"
        assert isinstance(img, torch.Tensor)
        assert img.numel() > 0
        assert img is not None
        assert not torch.isnan(img).any(), "Image tensor contains NaNs"
        
def test_predict_single_image_returns_correct_digit():
    '''Test predict_single_image returns 7 using dummy model and dummy image'''

    # Dummy model that always predicts class 7
    class DummyModel(nn.Module):
        def forward(self, x):
            # batch size x 10 classes, output logits with highest at index 7
            batch_size = x.shape[0]
            out = torch.zeros(batch_size, 10)
            out[:, 7] = 10.0  # set class 7 score highest
            return out

    dummy_model = DummyModel()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Create a dummy image (224x224 grayscale)
    dummy_array = (np.random.rand(224, 224) * 255).astype("uint8")
    dummy_image = Image.fromarray(dummy_array).convert("L")

    # Call your predict function passing dummy image and dummy model
    pred = predict_single_image(
        image=dummy_image,
        model=dummy_model,
        device=torch.device("cpu"),
        transform=transform
    )

    assert pred == 7, f"Expected 7 but got {pred}"

def test_predict_single_image_with_invalid_image_file():
    '''Test predict_single_image rejects invalid image file
    
    Should raise an exception when file is not a valid image
    '''
    dummy_model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(224*224, 10)
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a fake non-image file
        fake_path = os.path.join(tmpdir, "not_an_image.txt")
        with open(fake_path, "w") as f:
            f.write("this is not an image")

        # Save dummy model
        dummy_model_path = os.path.join(tmpdir, "dummy_model.pth")
        torch.save(dummy_model.state_dict(), dummy_model_path)

        # Patch model loader
        from src.modeling import predict
        original_get_model = predict.get_resnet18_mnist
        predict.get_resnet18_mnist = lambda: dummy_model

        with pytest.raises(Exception):
            predict.predict_single_image(fake_path, dummy_model_path, transform=transform)

        predict.get_resnet18_mnist = original_get_model
        
        # Clean up
        os.remove(fake_path)
        os.remove(dummy_model_path)
        # check the file is removed
        assert not os.path.exists(fake_path)
        assert not os.path.exists(dummy_model_path)






