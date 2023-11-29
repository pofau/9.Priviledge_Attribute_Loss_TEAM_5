import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

num_classes = 6  # AffectNet a 6 classes d'émotions
class VGGFace(nn.Module):
    def __init__(self, num_classes=6, pretrained=True, dropout_prob=0.6):
        super(VGGFace, self).__init__()

        # Utilise le modèle VGG16 pré-entraîné
        vgg16 = models.vgg16(pretrained=pretrained)

        # Supprime la dernière couche FC
        vgg16 = nn.Sequential(*list(vgg16.children())[:-1])

        # Freeze toutes les couches convolutionnelles
        for param in vgg16.parameters():
            param.requires_grad = False

        # Ajoute une nouvelle couche FC pour le nombre de classes dans AffectNet et RAF-DB
        self.features = vgg16
        self.fc6 = nn.Linear(512, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=dropout_prob)
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=dropout_prob)
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.dropout7(x)
        x = self.fc8(x)
        return x


# Definition du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGGFace(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Definition des transformations pour la normalisation des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Modèle VGGFace prend en entrée des images de taille 224x224
    transforms.ToTensor(),
])

# Definition des datasets
train_dataset = ImageFolder(root='path_to_your_dataset/train', transform=transform)
val_dataset = ImageFolder(root='path_to_your_dataset/validation', transform=transform)

# Define batch size
batch_size = 32

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Now, you can use train_loader and val_loader in your training loop

# Boucle d'entraînement
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Met le modèle en mode entraînement
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print average training loss for the epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    # Validation
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print validation accuracy and loss
    print(f"Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {100*correct/total}%")

print("Training finished")
