import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os

class RAFDBDataset(Dataset):
    def __init__(self, root_dir, label_dir, transform=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.transform = transform
        self.labels, self.image_paths = self._load_data()

    def _load_data(self):
        labels = []
        image_paths = []

        labels_file_path = os.path.join(self.label_dir, 'list_partition_label.txt')
        with open(labels_file_path, 'r') as file:
            lines = file.readlines()

            for line in lines:
                parts = line.strip().split(' ')
                label = int(parts[1])
                image_path = os.path.join(self.root_dir, 'aligned', parts[0])  # Change 'aligned' to the desired folder
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

root_dir = '../datasets/RAF-DB/Image/aligned/'
label_dir = '../datasets/RAF-DB/Image/aligned/'



# Assuming you have already defined full_dataset, train_subset, and test_subset
train_dataset = RAFDBDataset(root_dir="C:\Users\MCE30\Desktop\SAR\M2 SAR\MLA\Projet\RAF-DB\Image\aligned\train", label_dir = r'C:\Users\MCE30\Desktop\SAR\M2 SAR\MLA\Projet\RAF-DB\Image\aligned', transform=train_transform)
test_dataset = RAFDBDataset(root_dir=r'C:\Users\MCE30\Desktop\SAR\M2 SAR\MLA\Projet\RAF-DB\Image\aligned\test', label_dir = r'C:\Users\MCE30\Desktop\SAR\M2 SAR\MLA\Projet\RAF-DB\Image\aligned\list_partition_label.txt',transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("RAF-DB Dataset Loaded !")
