
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim

# L'optimiseur
learning_rate = 5e-5

# Époque :
num_epochs = 75  # Assurez-vous que cette variable est nommée correctement

# Modèle ResNet50 pré-entraîné
model = models.resnet50(pretrained=True)

num_classes = 6  # AffectNet a 6 classes d'émotions
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Définir l'optimiseur et la fonction de perte
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()  

#Boucle d'entraînelent
for epoch in range(num_epochs):
    for images, labels in data_loader:
        optimizer.zero_grad()
        
        outputs = model(images)

        # Calcule la perte classique
        loss = criterion(outputs, labels)

        # Pour l'instant, la perte PAL est mise à 0 car elle n'est pas encore implémentée
        pal_loss_value = 0  # à créer

        # Combinez les pertes
        total_loss = loss + pal_loss_value

        total_loss.backward()
        optimizer.step()
