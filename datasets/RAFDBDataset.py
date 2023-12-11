import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class RAFDBDataset(Dataset):
    def __init__(self, root_dir, label_dir, subset, label_file_name, transform=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.transform = transform
        self.subset = subset
        self.label_file_name = label_file_name
        self.labels, self.image_paths = self._load_data()

    def _load_data(self):
        labels = []
        image_paths = []
        
        labels_file_path = os.path.join(self.label_dir, self.label_file_name)
        with open(labels_file_path, 'r') as file:
            lines = file.readlines()

            for line in lines:
                parts = line.strip().split(' ')
                label = int(parts[1])
                image_path = os.path.join(self.root_dir, self.subset, parts[0])
                labels.append(label)
                image_paths.append(image_path)

        return labels, image_paths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# Transform only to tensor because images are already aligned in the dataset
transform = transforms.Compose([

    transforms.ToTensor(),
])

root_dir = '../datasets/RAF-DB/Image/aligned/'
label_dir = '../datasets/RAF-DB/Image/aligned/labels'



# Assuming you have already defined full_dataset, train_subset, and test_subset
train_dataset = RAFDBDataset(root_dir=root_dir, label_dir = label_dir, subset = 'train', label_file_name='train_label.txt', transform=transform)
test_dataset = RAFDBDataset(root_dir=root_dir, label_dir = label_dir, subset = 'test', label_file_name='test_label.txt', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

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

# Identifier la dernière couche de convolution
last_conv_layer = model[0].features[28]
print(last_conv_layer)
optimizer = optim.Adam(model.parameters(), lr=4e-5)

# Fonction pour enregistrer le gradient
def save_gradient(grad):
    global conv_output_gradient
    conv_output_gradient = grad

print("RAF-DB Dataset Loaded !")
