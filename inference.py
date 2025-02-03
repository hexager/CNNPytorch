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
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        # Update per-class counts and track misclassified examples
            for img, target, pred in zip(images.cpu(), targets.cpu(), preds.cpu()):
                target = int(target)
                pred = int(pred)
                class_counts[target] += 1
                class_predicted[pred] += 1
                if target == pred:
                    class_correct[target] += 1
                else:
                    # Save misclassified image example
                    misclassified_samples.append((img, pred, target))

    # -----------------------------------------------------
    # Overall Accuracy
    overall_accuracy = sum(class_correct) / len(all_targets)
    print("Overall Accuracy: {:.4f}".format(overall_accuracy))

    # -----------------------------------------------------
    # Per-class precision, recall and f1-score.
    precision = [0] * num_classes
    recall = [0] * num_classes
    f1_score = [0] * num_classes

    for i in range(num_classes):
        if class_predicted[i] > 0:
            precision[i] = class_correct[i] / class_predicted[i]
        else:
            precision[i] = 0

        if class_counts[i] > 0:
            recall[i] = class_correct[i] / class_counts[i]
        else:
            recall[i] = 0

        if precision[i] + recall[i] > 0:
            f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            f1_score[i] = 0

    # Compute macro F1 score (average of per-class F1 scores)
    macro_f1 = sum(f1_score) / num_classes

    # Display the computed metrics
    print("\nPer-class Metrics:")
    for i in range(num_classes):
        print("Class {}: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(i, precision[i], recall[i], f1_score[i]))
    print("\nMacro F1 Score: {:.4f}".format(macro_f1))

if __name__ == '__main__':
    run_inference()