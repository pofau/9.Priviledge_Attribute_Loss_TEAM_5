import torchvision.models as models
import torch.nn as nn

# modèle VGG16 pré-entraîné
vgg16_model = models.vgg16(pretrained=True)

# Adapte la dernière couche FC pour le nombre de classes dans AffectNet
num_classes = 6  # AffectNet a 6 classes d'émotions
vgg16_model.classifier[6] = nn.Linear(vgg16_model.classifier[6].in_features, num_classes)
