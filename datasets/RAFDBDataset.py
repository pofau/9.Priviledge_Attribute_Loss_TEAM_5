import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class RAFDBDataset(Dataset):
    def __init__(self, root_dir, label_path, subset, transform=None):
        self.root_dir = root_dir
        self.label_path = label_path
        self.transform = transform
        self.classes = sorted([f for f in os.listdir(os.path.join(root_dir, subset)) if os.path.isdir(os.path.join(root_dir, subset, f))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []

        with open(label_path, 'r') as file:
            lines = file.readlines()

            for line in lines:
                parts = line.strip().split(' ')
                file_name = parts[0]
                cls_name = parts[1]
                cls_idx = self.class_to_idx[cls_name]

                image_path = os.path.join(root_dir, subset, file_name)
                self.samples.append((image_path, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path, cls_idx = self.samples[idx]

        image = Image.open(file_path)

        if self.transform:
            image = self.transform(image)

        return image, cls_idx

    


# Transform function for training data
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),
])

# Transform function for test data
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Assuming you have already defined full_dataset, train_subset, and test_subset
train_dataset = RAFDBDataset(root_dir="C:\Users\MCE30\Desktop\SAR\M2 SAR\MLA\Projet\RAF-DB\Image\aligned", label_dir = r'C:\Users\MCE30\Desktop\SAR\M2 SAR\MLA\Projet\RAF-DB\Image\aligned\list_partition_label.txt', transform=train_transform)
test_dataset = RAFDBDataset(root_dir=r'C:\Users\MCE30\Desktop\SAR\M2 SAR\MLA\Projet\RAF-DB\Image\aligned', label_dir = r'C:\Users\MCE30\Desktop\SAR\M2 SAR\MLA\Projet\RAF-DB\Image\aligned\list_partition_label.txt',transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("RAF-DB Dataset Loaded !")
