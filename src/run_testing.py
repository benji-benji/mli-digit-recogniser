import torch
import sklearn as sk
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from modeling.train import get_mnist_dataloaders, get_resnet18_mnist

def evaluate_model(model_path="models/resnet18_mnist.pth", device=None):
    # Set device
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Load data
    _, test_loader = get_mnist_dataloaders(batch_size=128)

    # Load model
    model = get_resnet18_mnist().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Evaluate
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds


def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    labels, preds = evaluate_model()
    print(classification_report(labels, preds))
    plot_confusion_matrix(labels, preds)