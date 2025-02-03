import torch
from torchvision import transforms
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import json

# Import your CNN model - assuming you have it defined in model.py
from model import CNN

# Setup device and load your pretrained CNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("cnn_mnist.pth", map_location=device))
model.eval()
stats_path='stats.json'
def load_stats(stats_path='stats.json'):
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats['mean'], stats['std']
# Define transform to preprocess your custom images:
# - Resize to 28x28 (in case the image is not exactly 28x28)
# - Convert image to grayscale (ensuring single channel)
mean, std = load_stats(stats_path)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

# List your custom image file paths
image_paths = ["OutsideofData\\image1.png", "OutsideofData\\image2.png", "OutsideofData\\image3.png", "OutsideofData\\image4.png"]

# Holders for predictions and images for display
predictions = []
display_images = []

# Process each image, perform inference, and store original images for display
for path in image_paths:
    # Open the image using Pillow
    orig_image = Image.open(path)
    # Save a copy for display (convert to grayscale for consistency)
    display_images.append(orig_image.convert("L"))
    
    # Preprocess image using the transform
    image = transform(orig_image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    
    # Run inference
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True).item()
        predictions.append(pred)
plt.ion()
# Display images along with their predictions
num_images = len(display_images)
fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
if num_images == 1:
    axes = [axes]  # Ensure axes is iterable when there's only one image

for ax, img, pred in zip(axes, display_images, predictions):
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(f'Pred: {pred}')
plt.show(block=True)
