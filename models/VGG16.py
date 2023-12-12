import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, models
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
from datasets import load_dataset  # Assurez-vous que la bibliothèque `datasets` est installée

class AffectNetClassifier(nn.Module):
    def __init__(self, num_classes=7, train_size_ratio=0.2, batch_size=16, learning_rate=4e-5):
        super(AffectNetClassifier, self).__init__()
        # Charger le dataset
        full_dataset = load_dataset("Piro17/affectnethq", split='train')
        train_size = int(train_size_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_subset, test_subset = random_split(full_dataset, [train_size, test_size])

        # Définir les transformations
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation((-10, 10)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # Créer les datasets et les dataloaders
        self.train_loader = DataLoader(AffectNetHqDataset(Subset(full_dataset, train_subset.indices), transform=self.train_transform), 
                                       batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(AffectNetHqDataset(Subset(full_dataset, test_subset.indices), transform=self.test_transform), 
                                      batch_size=batch_size, shuffle=False)

        # Configuration du modèle VGG16
        self.model = models.vgg16(pretrained=True)
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        self.model.classifier.add_module('final_fc', nn.Linear(4096, num_classes))
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.model(x)

# Utilisation
classifier = AffectNetClassifier()
summary(classifier.model, (3, 224, 224))
