import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from PIL import Image
from gradcam import GradCAM  # Assurez-vous d'avoir le module GradCAM approprié installé
import numpy as np 
class AffectNetHqDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']

        if self.transform:
            image = self.transform(image)

        return image, label

# Load the full dataset
full_dataset = load_dataset("Piro17/affectnethq", split='train')

# Split the dataset into train and test subsets
train_size = int(0.01 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_subset, test_subset = random_split(full_dataset, [train_size, test_size])

# Define transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation((-10, 10)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create the dataset and dataloader using the subsets
train_dataset = AffectNetHqDataset(Subset(full_dataset, train_subset.indices), transform=train_transform)
test_dataset = AffectNetHqDataset(Subset(full_dataset, test_subset.indices), transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

import torch.nn as nn
import matplotlib.pyplot as plt

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv_layers = nn.Sequential(
            # Premier bloc
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Deuxième bloc
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Troisième bloc
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # Ajout de BatchNorm
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Quatrième bloc
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # Ajout de BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # Ajout de BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # Ajout de BatchNorm
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Cinquième bloc
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # Ajout de BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # Ajout de BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # Ajout de BatchNorm
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.last_conv_layer = self.conv_layers[-1]
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 1024),  
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 7),
            nn.Softmax(dim=1)
        )

        # Ajout d'un attribut pour stocker les gradients de la dernière couche de convolution
        self.last_conv_gradients = None

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

    def capture_last_conv_gradients(self):
        return self.last_conv_layer


# Création du modèle VGG16
vgg16 = VGG16()
print(vgg16)

# Récupération de la dernière couche du modèle VGG16
target_layer = vgg16.conv_layers[-1]  # Dernière couche de convolutions

# Assurez-vous que toutes les dépendances sont importées
import time
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

# Paramètres
num_epochs = 5  # Normalement 75 epochs
optimizer = torch.optim.Adam(vgg16.parameters(), lr=4e-5)
DEVICE = "cpu"  # "cuda" ou "cpu"
max_iterations = 40000
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

train_losses = []
train_accuracies = []

# Paramètres
num_epochs = 5
optimizer = optim.Adam(vgg16.parameters(), lr=4e-5)
criterion = nn.CrossEntropyLoss()
DEVICE = "cpu"  # "cuda" ou "cpu"
max_iterations = 40000

# Entraînement
for epoch in range(num_epochs):
    start = time.perf_counter()
    vgg16.train()
    running_loss = 0.0
    correct_pred = 0
    total_samples = 0

    for i, (data, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)):
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        data.requires_grad = True  # Enable gradient tracking for input

        # Forward pass
        outputs = vgg16(data)

        # Calcul de la perte standard
        standard_loss = criterion(outputs, targets)

        optimizer.zero_grad()

        # Ajoutez le hook à la dernière couche de convolution
        last_conv_layer = vgg16.last_conv_layer
        hook = last_conv_layer.register_forward_hook(lambda m, inp, out: None)  # Créez une fonction vide pour le hook
        output = last_conv_layer(data)  # Passez les données par la dernière couche de convolution

        # Calcul du grad*input
        grad_input = torch.abs(output)  # Gradients de la dernière couche de convolution

        # Calculez grad_input en spécifiant également retain_graph=True
        grad_input.backward(gradient=torch.ones_like(output), retain_graph=True)

        # Maintenant, grad_input contient les contributions de chaque pixel
        grad_input = data.grad  # Obtenez les gradients de l'entrée par rapport à la sortie

        # Visualisez grad_input (contributions) pour chaque image dans le batch
        for j in range(data.size(0)):
            input_image = data[j].cpu().detach().numpy().transpose((1, 2, 0))  # Convert image tensor to NumPy array
            grad_input_image = grad_input[j].abs().cpu().detach().numpy().transpose((1, 2, 0))  # Convert grad_input tensor to NumPy array

            # Affichez l'image d'entrée et les contributions
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(input_image)
            plt.title("Input Image")

            plt.subplot(1, 2, 2)
            plt.imshow(grad_input_image.mean(axis=-1), cmap='viridis')
            plt.title("Grad*Input (Contributions)")

            plt.show()

        hook.remove()  # Désenregistrez le hook après utilisation

        # Backward pass for standard loss
        standard_loss.backward()  # Retirez retain_graph=True

        # Zero gradients for the next iteration
        optimizer.zero_grad()

        # Optimisation
        optimizer.step()

        # Mise à jour des métriques
        running_loss += standard_loss.item()
        _, pred = torch.max(outputs, 1)
        correct_pred += (pred == targets).sum().item()
        total_samples += targets.size(0)

        if i >= max_iterations:
            break

    # Affichage des statistiques après chaque époque
    epoch_loss = running_loss / (i + 1)
    epoch_accuracy = 100 * correct_pred / total_samples
    end = time.perf_counter()
    print(f'Epoch {epoch + 1}/{num_epochs}\tTrain loss: {epoch_loss:.4f}\tTrain accuracy: {epoch_accuracy:.2f}%')
    print(f'Time: {end - start:.2f}s')

print('Finished training!')


