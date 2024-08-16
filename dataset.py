import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# Placeholder for a dataset class
class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self):
        # Load your dataset here
        pass
    
    def __len__(self):
        return 100  # example length
    
    def __getitem__(self, idx):
        # Return a single item as (feature, label)
        return torch.randn(100, 128), torch.randint(0, 29, (100,))  # Example data
