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

# Définir les transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Créer le dataset et le dataloader
affectnet_dataset = AffectNetHqDataset(transform=transform)
data_loader = DataLoader(affectnet_dataset, batch_size=16, shuffle=False)

# Charger le modèle pré-entraîné VGG16
base_model = torchvision.models.vgg16(pretrained=True)
# Supprimer la dernière couche entièrement connectée
base_model.classifier = nn.Sequential(*list(base_model.classifier.children())[:-1])

# Ajouter une nouvelle couche adaptée à 7 classes
num_classes = 7
classifier_layer = nn.Linear(4096, num_classes)
model = nn.Sequential(base_model, classifier_layer)

# Afficher la structure du modèle
summary(model, (3, 224, 224))  # Assurez-vous d'ajuster les dimensions en fonction de vos données

optimizer = optim.Adam(model.parameters(), lr=4e-5)
num_epochs = 10
criterion = torch.nn.CrossEntropyLoss()
loss_values = [] 
accuracy_values = []  
lr = 4e-5
power = 5

def adjust_learning_rate(optimizer, epoch, num_epochs, initial_lr, power):
    """Ajuste le taux d'apprentissage selon une politique de décroissance polynomiale."""
    lr = initial_lr * (1 - (epoch / num_epochs)) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


for epoch in range(num_epochs):
    adjust_learning_rate(optimizer, epoch, num_epochs, lr, power)  # Mise à jour du taux d'apprentissage
    model.train()
    running_loss = 0.0
    running_pal_loss = 0.0
    running_corrects = 0.0
    total_samples = 0.0
    for images, labels in tqdm(train_loader):
        
        # Initialiser un tenseur pour stocker toutes les heatmaps
        batch_heatmaps = generate_batch_heatmaps(images, heatmap_generator)
            
         # Ensure that images require gradients
        images.requires_grad_()

         # Forward pass
        outputs = model(images)
        labels = labels.long()

        # Calcul de la classification loss
        classification_loss = criterion(outputs, labels)

        # Backward pass for gradients with respect to the input images
        classification_loss.backward(retain_graph=True)  
        gradients = images.grad

        # Compute the attribution maps as the element-wise product of the gradients and the input images
        attribution_maps = gradients * images

        # Compute the PAL loss using the attribution maps and the prior maps
        pal_loss_fn = PrivilegedAttributionLoss()
        pal_loss = pal_loss_fn(attribution_maps, batch_heatmaps)

        # Calcul de la PAL loss et de la classification loss
        total_loss = classification_loss + pal_loss

        # Backpropagation et optimisation
        optimizer.zero_grad()  # Clear gradients before the backward pass
        total_loss.backward()
        optimizer.step()

        # Mise à jour des running loss et PAL loss
        running_loss += classification_loss.item()
        running_pal_loss += pal_loss.item()         

        #if epoch == 0:
         #   plot_element(images, batch_heatmaps, attribution_maps, gradients, 0)

        # Mise à jour des running loss et PAL loss
        running_loss += classification_loss.item()
        running_pal_loss += pal_loss.item()

        # Calcul de l'accuracy
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += labels.size(0)

    # Calcul des moyennes pour l'époque
    epoch_loss = running_loss / len(train_loader)
    epoch_pal_loss = running_pal_loss / len(train_loader)
    epoch_acc = running_corrects.double() / total_samples

    # Ajouter les valeurs moyennes aux listes
    loss_values.append(epoch_loss)
    accuracy_values.append(epoch_acc)

    # Affichage des résultats pour l'époque
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print(f'Loss: {epoch_loss:.4f}, PAL Loss: {epoch_pal_loss:.4f}, Accuracy: {epoch_acc:.4f}')

import matplotlib.pyplot as plt

# Liste de longueurs de vecteur que vous souhaitez utiliser
vector_lengths = np.linspace(0, len(loss_values), len(loss_values))

# Plot de la perte en fonction de la longueur du vecteur
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(vector_lengths, loss_values, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Perte')
plt.title("Loss en fonction de l'epoch")

# Plot de la précision en fonction de la longueur du vecteur
plt.subplot(1, 2, 2)
plt.plot(vector_lengths, accuracy_values, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title("Accuracy en fonction de l'epoch")

plt.tight_layout()  # Pour éviter que les titres se chevauchent
plt.show()
