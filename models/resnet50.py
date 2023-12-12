import torchvision.models as models
import torchvision
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim

class CustomResNet(nn.Module):
    def __init__(self, num_classes=7):
        super(CustomResNet, self).__init__()
        # Charger le modèle pré-entraîné ResNet50
        base_model = torchvision.models.resnet50(pretrained=True)
        # Supprimer la dernière couche entièrement connectée (fc)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        # Ajouter une nouvelle couche adaptée à num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )
        # Ajouter le hook sur la dernière couche de convolution
        self.gradients = None  # Stocke les gradients de la dernière couche de convolution
        self._add_conv_hook()

    def _add_conv_hook(self):
        # ... [méthode pour ajouter un hook à la dernière couche de convolution]
        last_conv_layer = list(self.features.children())[-3][2].conv3
        last_conv_layer.register_full_backward_hook(self._hook_fn)

    def _hook_fn(self, module, grad_input, grad_output):
        # Enregistre les gradients de la dernière couche de convolution
        self.gradients = grad_output[0]

    def get_activations_gradient(self):
        # Renvoie les gradients enregistrés
        return self.gradients

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Utilisation de la classe
num_classes = 7
model = CustomResNet(num_classes)

# Afficher la structure du modèle
summary(model, (3, 224, 224))

# Définir l'optimiseur
learning_rate = 5e-5  # Assurez-vous de définir la valeur de learning_rate
optimizer = optim.Adam(model.parameters(), learning_rate)
