# Import Libraries
import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define Transform: Convert to Tensor and Normalize
transform = transforms.Compose([
    transforms.ToTensor(),                             # Converts to tensor (range 0-1)
    transforms.Normalize((0.5,), (0.5,))               # Normalize to range [-1, 1]
])

# Load Fashion MNIST Dataset (Train Set)
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# Class Labels in Fashion MNIST
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Extract a few sample images and their labels
figure = plt.figure(figsize=(8,8))
for i in range(9):
    img, label = train_dataset[i]                      # Get image and label
    img = img.squeeze()                                # Remove channel dimension (1,28,28) -> (28,28)
    plt.subplot(3,3,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(classes[label])
    plt.axis('off')

plt.tight_layout()
plt.savefig('fashion_mnist_samples_19-06-2025.png')
plt.show()
