import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN
import matplotlib.pyplot as plt

def load_stats(stats_path='stats.json'):
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats['mean'], stats['std']

def run_inference(model_path='cnn_mnist.pth', stats_path='stats.json'):
    # Load normalization stats computed during training
    mean, std = load_stats(stats_path)
    print(f"Loaded Mean: {mean:.4f}  Std: {std:.4f}")
    
    # Define transform using loaded stats
    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    
    # Load the MNIST test dataset using normalization computed from the training set
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=final_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    
    # Run inference on 10 test samples
    all_preds = []
    all_targets = []
    num_classes = 10
    class_counts = [0] * num_classes        # Number of times each class appears (ground truth)
    class_correct = [0] * num_classes       # True positives for each class
    class_predicted = [0] * num_classes     # Number of times each class was predicted
    misclassified_samples = []
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image = image.to(device)
            output = model(image)
            prediction = output.argmax(dim=1, keepdim=True)
            print(f"Predicted: {prediction.item()}  |  Actual: {label.item()}")
            if i == 9:
                break

if __name__ == '__main__':
    run_inference()