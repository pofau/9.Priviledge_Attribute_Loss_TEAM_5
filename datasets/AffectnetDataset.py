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

if __name__ == '__main__':

    # Define the transformation to apply to the data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create the dataset
    affectnet_dataset = AffectNetHqDataset(transform=transform)
    data_loader = DataLoader(affectnet_dataset, batch_size=16, shuffle=False)


