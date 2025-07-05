import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os

class ImageDataset(Dataset):
    def __init__(self, df, root_dir, processor, tokenizer):
        self.df = dataframe
        self.root_dir = root_dir
        
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        
        image = 