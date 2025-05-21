import torch
import torchvision.transforms as transforms
from PIL import Image
from src.modeling.train import get_resnet18_mnist


def predict_single_image(image, model, device=None, transform=None):
    """Predict the digit in a single image using a trained ResNet18 model.

    arguments:
        image_path (PIL.Image or ndarray): Input image to predict.
        model_path (str): Path to the trained model weights.
        device (torch.device, optional): Device to use ('cpu' or 'cuda'). If None, auto-detect.

    returns:
        int (predicted digit 1-9) 
    """
    # Set device
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if transform is None:
        # Default transform if not provided
        transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])

    # load and preprocess image 
    # If input is a path, open image, else assume PIL.Image or ndarray
    if isinstance(image, str):
        img = Image.open(image).convert("L")
    elif isinstance(image, Image.Image):
        img = image.convert("L")
    else:
        # If ndarray, convert to PIL Image first
        img = Image.fromarray(image).convert("L")
    
    
    img = transform(img)  # transform: resize, tensor, normalize
    img = img.unsqueeze(0)  # Add batch dimension: [1, 1, 224, 224]
    img = img.to(device)
    # load model
    model = model.to(device)
    #model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # set model to evaluation mode

    # predict
    with torch.no_grad():
         # stop pytorch tracking changes in gradient
         # because we are no longer training, we are now predicting 
        #img = img.to(device) # move img to device 
        output = model(img) # forward pass
        # get model output
        pred = output.argmax(dim=1).item()  # get predicted class

    return pred
