from PIL import Image
import os
import pandas
import torch
from torch.utils.data import Dataset

class ICDAR2013(Dataset):
    
    BASE_URL = os.path.join('data', 'icdar2013')
    
    def __init__(self, processor, tokenizer, train = True, max_target_length=128):
        super().__init__()
        
        data_url = 'train_data.csv' if train else 'test_data.csv'
        self.data = pandas.read_csv(os.path.join(ICDAR2013.BASE_URL, data_url), encoding = 'utf-8', engine='python')
        self.sub_folder = 'train_cropped_images' if train else 'test_cropped_images'
        
        self.processor = processor
        self.tokenizer = tokenizer  
        
        self.max_target_length = max_target_length
                                    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        
        file_name = self.data['file_name'][idx]
        text = self.data['text'][idx]
        path = os.path.join(ICDAR2013.BASE_URL, self.sub_folder, file_name)
        
        image = Image.open(path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        labels = self.tokenizer(text,
                                padding="max_length",
                                max_length=self.max_target_length,
                                )
    
        input_ids = labels.input_ids
        attention_masks = labels.attention_mask
        
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in input_ids]
        
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels), "decoder_attention_mask": torch.tensor(attention_masks)}
        return encoding