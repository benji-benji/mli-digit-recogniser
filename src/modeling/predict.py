import torch
import torchvision.transforms as transforms
from PIL import Image
from src.modeling.train import get_resnet18_mnist


def predict_single_image(image_path, model_path="models/resnet18_mnist.pth", device=None, transform=None):
    """Predict the digit in a single image using a trained ResNet18 model.

    arguments:
        image_path (str): Path to the image file.
        model_path (str): Path to the trained model weights.
        device (torch.device, optional): Device to use ('cpu' or 'cuda'). If None, auto-detect.

    returns:
        int (predicted digit 1-9) 
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if transform is None:
        # Default transform if not provided
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    # load and preprocess image 
    img = Image.open(image_path).convert("L")  # open and make grayscale 
    img = transform(img)  # transform: resize, tensor, normalize
    img = img.unsqueeze(0)  # Add batch dimension: [1, 1, 224, 224]

    # load model
    model = get_resnet18_mnist().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # set model to evaluation mode

    # predict
    with torch.no_grad():
         # stop pytorch tracking changes in gradient
         # because we are no longer training, we are now predicting 
        img = img.to(device) # move img to device 
        output = model(img) # forward pass
        # get model output
        pred = output.argmax(dim=1).item()  # get predicted class

    return pred
