from PIL import Image
import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms

class IndicSTR(Dataset):
    def __init__(self, root_dir, df, processor, tokenizer, max_target_length=256, augment_images=False):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.image_size = (self.processor.size['width'], self.processor.size['height'])
        self.augment_images = augment_images
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=2),
            transforms.RandomResizedCrop(size=self.image_size, scale=(0.9, 1.0)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor()
        ])


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        items = self.df.iloc[idx]
        if len(items) == 2:
            file_name, text = items
        else:
            file_name, text, _ = items

               
        # prepare image (i.e. resize + normalize)
        image = Image.open(os.path.join(self.root_dir, file_name)).convert("RGB")
        if self.augment_images:
            image = self.augmentation_transforms(image)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Apply geometric and photometric augmentations using torchvision transforms

        labels = self.tokenizer(text,
                                add_special_tokens=False,
                                # padding=True,
                                # return_tensors="pt",
                                padding="max_length",
                                max_length=self.max_target_length,
                                )
        
        input_ids = labels.input_ids
        attention_masks = labels.attention_mask
        # important: make sure that PAD tokens are ignored by the loss function
        input_ids = [label if label != self.tokenizer.pad_token_id else -100 for label in input_ids]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(input_ids), "decoder_attention_mask": torch.tensor(attention_masks)}
        return encoding
