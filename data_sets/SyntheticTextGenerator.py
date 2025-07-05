import os
import random
import power_text
from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SyntheticDataset(Dataset):
    
    def __init__(self, 
                 corpus_dir, 
                 fonts_dir, 
                 processor,
                 tokenizer,
                 max_target_length=256,
                 target_shape = (256, 256), 
                 backgrounds_dir=None, 
                 size=10000, 
                 transform=None):
        """
        PyTorch Dataset for on-the-fly synthetic text image generation for STR.
        
        Args:
            corpus_dir (str): Directory with corpus files (e.g., hindi_corpus.txt).
            fonts_dir (str): Directory with .ttf/.otf fonts.
            backgrounds_dir (str, optional): Directory with background images.
            size (int): Number of samples to generate (controls dataset length).
            transform (callable, optional): Transform to apply to images.
        """
        self.corpus_dir = corpus_dir
        self.fonts_dir = fonts_dir
        self.backgrounds_dir = backgrounds_dir
        self.size = size
        self.transform = transform
        self.image_size = (256, 256)  # (width, height) for 256x256
        self.target_size = target_shape  # (height, width) for 64x128

        # Load corpora
        self.corpora = {}
        for corpus_file in os.listdir(corpus_dir):
            if corpus_file.endswith(".txt"):
                lang = corpus_file.replace("_corpus.txt", "")
                with open(os.path.join(corpus_dir, corpus_file), "r", encoding="utf-8") as f:
                    self.corpora[lang] = [line.strip() for line in f if line.strip()]
        
        # Load fonts
        self.fonts = {
            # "hindi": ImageFont.truetype(os.path.join(fonts_dir, "Mangal.ttf"), 24),
            # "tamil": ImageFont.truetype(os.path.join(fonts_dir, "Latha.ttf"), 24),
            "english": ImageFont.truetype(os.path.join(fonts_dir, "ARIALN.ttf"), 36),
        }
        
        # Load background images (if provided)
        self.backgrounds = []
        if backgrounds_dir and os.path.exists(backgrounds_dir):
            self.backgrounds = [
                os.path.join(backgrounds_dir, f) for f in os.listdir(backgrounds_dir)
                if f.endswith((".jpg", ".png"))
            ]
            
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
    
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        
    
    def __len__(self):
        """Return the dataset size."""
        return self.size
    
    def __getitem__(self, idx):
        """
        Generate a synthetic text image and label.
        
        Args:
            idx (int): Index (used for reproducibility with seed).
        
        Returns:
            tuple: (image, label) where image is a tensor and label is the text.
        """
        image, text, lang = self.get_data_point(idx)
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        labels = self.tokenizer(f"<2{lang}>{text}</s>",
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
        

    
    def get_data_point(self, idx):
        """
        Generate a synthetic text image and label.
        
        Args:
            idx (int): Index (used for reproducibility with seed).
        
        Returns:
            tuple: (image, label) where image is a tensor and label is the text.
        """
        # Set random seed for reproducibility
        # random.seed(idx)
        
        # Randomly select language and text
        lang = random.choice(list(self.corpora.keys()))
        text = random.choice(self.corpora[lang])
        font = self.fonts.get(lang, self.fonts["english"])  # Fallback to English font
        
        # Create temporary image for bounding box calculation
        temp_img = Image.new("RGB", (256, 256), (255, 255, 255))
        draw = ImageDraw.Draw(temp_img)
        bbox = draw.textbbox((0, 0), text.strip(), font=font)
        text_width = bbox[2] - bbox[0] + 50  # Add padding
        text_height = bbox[3] - bbox[1] + 50
        
        # Create image sized to text (with background)
        if self.backgrounds and random.random() < 0.5:
            bg_path = random.choice(self.backgrounds)
            img = Image.open(bg_path).convert("RGB")
            img = img.resize((max(text_width, img.size[0]), max(text_height, img.size[1])), Image.LANCZOS)
        else:
            img = Image.new("RGB", (text_width, text_height), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        
        # Draw text using PowerText
        power_text.draw_text(
            img,
            (10, 10),
            text.strip(),
            [power_text.Font(font, lambda _: True)],
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            max_x=text_width - 10,
            max_y=text_height - 10,
            has_emoji=False,
        )
        
        # Crop to text bounding box
        img = img.crop((0, 0, text_width, text_height))
        
        # Apply augmentations
        img = self.apply_augmentations(img)
        
        # Resize/pad to 64x128
        img = img.resize(self.target_size, Image.LANCZOS)
        
        # Apply transform
        img_tensor = self.transform(img)
        
        return img_tensor, text, lang
    
    def apply_augmentations(self, image):
        """Apply random augmentations (blur, noise) to the image."""
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Random Gaussian blur
        if random.random() < 0.5:
            ksize = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        
        # # Random noise
        # if random.random() < 0.5:
        #     noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        #     image = cv2.add(image, noise)
        
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
