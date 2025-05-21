from src.modeling.train import train_model
import torch

device = None

if __name__ == "__main__":
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    train_model(epochs=12, batch_size=128, device=device)