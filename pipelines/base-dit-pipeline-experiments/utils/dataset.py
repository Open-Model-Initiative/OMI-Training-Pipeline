import torch
from torch.utils.data import Dataset
class ImagePromptDataset(Dataset):
    def __init__(self, dataset, transform=None, device = torch.device("cuda")):
        self.dataset = dataset
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        image = sample['image']
        prompt = sample['en_text']## for testing
        
        if self.transform:
            image = self.transform(image).to(self.device)
        
        return image, prompt