import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from PIL import Image

class AffectNetHqDataset(Dataset):
    def __init__(self, dataset, transform=None):
        # 'dataset' is now a subset of the original dataset
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
train_size = int(0.8 * len(full_dataset))
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

import matplotlib.pyplot as plt
import torchvision

def show_tensor_image(tensor):
    """Affiche une image tensorielle."""
    image = tensor.numpy().transpose((1, 2, 0))  # Convertir le tensor en array numpy et ajuster les dimensions
    plt.imshow(image)
    plt.show()

# Afficher la première image de l'ensemble d'entraînement
first_train_image, _ = train_dataset[0]
show_tensor_image(first_train_image)

# Afficher la dernière image de l'ensemble d'entraînement
last_train_image, _ = train_dataset[len(train_dataset) - 1]
show_tensor_image(last_train_image)

# Afficher la première image de l'ensemble de test
first_test_image, _ = test_dataset[0]
show_tensor_image(first_test_image)

# Afficher la dernière image de l'ensemble de test
last_test_image, _ = test_dataset[len(test_dataset) - 1]
show_tensor_image(last_test_image)

import torchvision.models as models
import torch
import torch.nn as nn
from torchsummary import summary

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv_layers = nn.Sequential(
            # Premier bloc
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Ajout de BatchNorm
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Ajout de BatchNorm
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Deuxième bloc
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # Ajout de BatchNorm
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # Ajout de BatchNorm
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Troisième bloc
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # Ajout de BatchNorm
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # Ajout de BatchNorm
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # Ajout de BatchNorm
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Quatrième bloc
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # Ajout de BatchNorm
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # Ajout de BatchNorm
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # Ajout de BatchNorm
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Cinquième bloc
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # Ajout de BatchNorm
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # Ajout de BatchNorm
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # Ajout de BatchNorm
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            # Couches entièrement connectées
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 7),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)  
        x = self.fc_layers(x)
        return x

# Création du modèle VGG16
vgg16 = VGG16()
summary(vgg16, (3, 224, 224))

from tqdm import tqdm
from random import sample


def adjust_learning_rate(optimizer, epoch, base_lr, max_epochs, power=1.0):
    lr = base_lr * (1 - epoch / max_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
# Fonction de perte et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg16.parameters(), lr=4e-5)


# Boucle d'entraînement
num_epochs = 75  # Définir le nombre d'époques

train_losses = []
train_accuracies = []

batch_size = 16


for epoch in range(num_epochs):
    adjust_learning_rate(optimizer, epoch, base_lr=4e-5, max_epochs=num_epochs)
    vgg16.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = vgg16(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    

    epoch_loss = running_loss / total  # Assurez-vous que ce calcul est correct
    epoch_accuracy = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
