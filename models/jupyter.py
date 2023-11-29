import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class AffectNetHqDataset(Dataset):
    def __init__(self, split='train', transform=None):
        self.dataset = load_dataset("Piro17/affectnethq", split=split)
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


# Définir les transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Créer le dataset et le dataloader
affectnet_dataset = AffectNetHqDataset(transform=transform)
data_loader = DataLoader(affectnet_dataset, batch_size=16, shuffle=False)

#%%

import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# Charger un modèle VGG16 pré-entraîné
vgg16_model = models.vgg16(pretrained=True)

# Adapter la dernière couche FC pour le nombre de classes dans AffectNet
num_classes = 6  # AffectNet a 6 classes d'émotions
vgg16_model.classifier[6] = nn.Linear(vgg16_model.classifier[6].in_features, num_classes)

# Fonction de perte et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg16_model.parameters(), lr=4e-5)

# Boucle d'entraînement
num_epochs = 75

for epoch in range(num_epochs):
    vgg16_model.train()
    running_loss = 0.0

    for images, labels in data_loader:
        optimizer.zero_grad()
        outputs = vgg16_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(data_loader)}")
#%%

# modèle VGG16 pré-entraîné
vgg16_model = models.vgg16(pretrained=True)

# Adapte la dernière couche FC pour le nombre de classes dans AffectNet
num_classes = 6  # AffectNet a 6 classes d'émotions
vgg16_model.classifier[6] = nn.Linear(vgg16_model.classifier[6].in_features, num_classes)


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv_layers = nn.Sequential(
            # Premier bloc
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Deuxième bloc
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Troisième bloc
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Quatrième bloc
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Cinquième bloc
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            # Couches entièrement connectées
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Création du modèle VGG16
vgg16 = VGG16()
