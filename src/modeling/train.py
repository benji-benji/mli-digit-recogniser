from torchvision.models import resnet18
import torch.nn as nn

def get_resnet18_mnist():
    model = resnet18(num_classes=10)  # 10 classes for MNIST
    # Change first conv layer to accept 1 channel (instead of 3 for RGB)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model
