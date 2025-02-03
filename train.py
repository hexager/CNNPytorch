# train.py
import torch
import json
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNN

def compute_stats(dataset):
    """
    Compute the global mean and std of the dataset.
    Expects dataset with transform=ToTensor() only.
    """
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data, _ = next(iter(loader))  # data shape: [N, 1, 28, 28]
    mean = data.mean().item()
    std = data.std().item()
    return mean, std

def train_model(epochs=5, batch_size=64, learning_rate=0.001, 
                model_save_path='cnn_mnist.pth', stats_path='stats.json'):
    
    # Load dataset with only ToTensor so we can compute stats
    base_transform = transforms.ToTensor()
    train_dataset_for_stats = datasets.MNIST(root='./data', train=True, download=True, transform=base_transform)
    
    # Compute mean and std
    mean, std = compute_stats(train_dataset_for_stats)
    print(f"Computed Mean: {mean:.4f}  Std: {std:.4f}")
    
    # Save these stats to a file for later use in inference
    with open(stats_path, 'w') as f:
        json.dump({'mean': mean, 'std': std}, f)
    
    # Define final transform using computed statistics
    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    
    # Reload training dataset with the final transform (normalization)
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=final_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()         # reset gradients
            outputs = model(images)       # forward pass
            loss = criterion(outputs, labels)
            loss.backward()               # backpropagation
            optimizer.step()              # weight update
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
    
    # Save the model state
    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")

if __name__ == '__main__':
    train_model()
    