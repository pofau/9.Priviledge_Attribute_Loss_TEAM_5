import torchvision.models as models
import torchvision
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim

class CustomResNet(nn.Module):
    def __init__(self, num_classes=7):
        super(CustomResNet, self).__init__()
        # Load the pretrained ResNet50 model
        base_model = torchvision.models.resnet50(pretrained=True)
        # Remove the last fully connected layer (fc)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        # Add a new layer adapted to num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )
        # Add hook to the last convolutional layer
        self.gradients = None  # Stores the gradients of the last convolutional layer
        self._add_conv_hook()

    def _add_conv_hook(self):
        # ... [method to add a hook to the last convolutional layer]
        last_conv_layer = list(self.features.children())[-3][2].conv3
        last_conv_layer.register_full_backward_hook(self._hook_fn)

    def _hook_fn(self, module, grad_input, grad_output):
        # Stores the gradients of the last convolutional layer
        self.gradients = grad_output[0]

    def get_activations_gradient(self):
        # Returns the stored gradients
        return self.gradients

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # Using the class
    num_classes = 7
    model = CustomResNet(num_classes)

    # Display the model structure
    summary(model, (3, 224, 224))

    # Define the optimizer
    learning_rate = 5e-5  # Make sure to set the learning_rate value
    optimizer = optim.Adam(model.parameters(), learning_rate)
