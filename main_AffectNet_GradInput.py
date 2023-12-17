# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import cv2
import os
import random
import glob
import face_recognition
from matplotlib.colors import LinearSegmentedColormap
from datasets.AffectnetDataset import AffectNetHqDataset
from datasets.RAFDBDataset import RAFDBDataset
from models.pal import PrivilegedAttributionLoss
from models.resnet50 import CustomResNet
from models.VGG16 import AffectNetClassifier
from utils.heatmap import *
from utils.plot_element import *

# Define the number of epochs
num_epochs = 20
# Define the batch size
batch_size = 16
# Define the input shape
input_shape = (224, 224)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create the dataset and dataloader
affectnet_dataset = AffectNetHqDataset(transform=transform)
data_loader = DataLoader(affectnet_dataset, batch_size=16, shuffle=False)

# Load the pretrained VGG16 model
base_model = torchvision.models.vgg16(pretrained=True)
# Remove the last fully connected layer
base_model.classifier = nn.Sequential(*list(base_model.classifier.children())[:-1])

# Add a new layer suitable for 7 classes
num_classes = 7
classifier_layer = nn.Linear(4096, num_classes)
model = nn.Sequential(base_model, classifier_layer)

# Display the model structure
summary(model, (3, 224, 224))  # Make sure to adjust the dimensions according to your data

optimizer = optim.Adam(model.parameters(), lr=4e-5)
num_epochs = 10
criterion = torch.nn.CrossEntropyLoss()
loss_values = [] 
accuracy_values = []  
lr = 4e-5
power = 5

def adjust_learning_rate(optimizer, epoch, num_epochs, initial_lr, power):
    """Adjust the learning rate according to a polynomial decay policy."""
    lr = initial_lr * (1 - (epoch / num_epochs)) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(num_epochs):
    adjust_learning_rate(optimizer, epoch, num_epochs, lr, power)  # Update the learning rate
    model.train()
    running_loss = 0.0
    running_pal_loss = 0.0
    running_corrects = 0.0
    total_samples = 0.0
    for images, labels in tqdm(train_loader):
        
        # Initialize a tensor to store all the heatmaps
        batch_heatmaps = generate_batch_heatmaps(images, heatmap_generator)
            
        # Ensure that images require gradients
        images.requires_grad_()

        # Forward pass
        outputs = model(images)
        labels = labels.long()

        # Calculate the classification loss
        classification_loss = criterion(outputs, labels)

        # Backward pass for gradients with respect to the input images
        classification_loss.backward(retain_graph=True)  
        gradients = images.grad

        # Compute the attribution maps as the element-wise product of the gradients and the input images
        attribution_maps = gradients * images

        # Compute the PAL loss using the attribution maps and the prior maps
        pal_loss_fn = PrivilegedAttributionLoss()
        pal_loss = pal_loss_fn(attribution_maps, batch_heatmaps)

        # Calculate the PAL loss and classification loss
        total_loss = classification_loss + pal_loss

        # Backpropagation and optimization
        optimizer.zero_grad()  # Clear gradients before the backward pass
        total_loss.backward()
        optimizer.step()

        # Update running loss and PAL loss
        running_loss += classification_loss.item()
        running_pal_loss += pal_loss.item()         

        # Update running loss and PAL loss
        running_loss += classification_loss.item()
        running_pal_loss += pal_loss.item()

        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += labels.size(0)

    # Calculate averages for the epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_pal_loss = running_pal_loss / len(train_loader)
    epoch_acc = running_corrects.double() / total_samples

    # Add the average values to the lists
    loss_values.append(epoch_loss)
    accuracy_values.append(epoch_acc)

    # Display results for the epoch
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print(f'Loss: {epoch_loss:.4f}, PAL Loss: {epoch_pal_loss:.4f}, Accuracy: {epoch_acc:.4f}')

import matplotlib.pyplot as plt

# List of vector lengths you want to use
vector_lengths = np.linspace(0, len(loss_values), len(loss_values))

# Plot the loss as a function of vector length
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(vector_lengths, loss_values, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Loss vs Epoch")

# Plot accuracy as a function of vector length
plt.subplot(1, 2, 2)
plt.plot(vector_lengths, accuracy_values, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title("Accuracy vs Epoch")

plt.tight_layout()  # To avoid overlapping titles
plt.show()

model.eval()  # Mettre le modèle en mode évaluation
test_running_corrects = 0.0
test_total_samples = 0.0

with torch.no_grad():  # Désactive le calcul du gradient
        for test_images, test_labels in tqdm(test_loader):
            test_outputs = model(test_images)
            _, test_preds = torch.max(test_outputs, 1)
            test_running_corrects += torch.sum(test_preds == test_labels.data)
            test_total_samples += test_labels.size(0)
# Calcul de l'accuracy de test pour l'époque
test_epoch_acc = test_running_corrects.double() / test_total_samples
test_accuracy_values.append(test_epoch_acc)

# Afficher les résultats de test pour l'époque
print(f'Test Accuracy: {test_epoch_acc:.4f}')
