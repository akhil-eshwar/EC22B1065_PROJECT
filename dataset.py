import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class LiTSDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        
        # Look for PNG files
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        self.masks_list = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])
        
        # Create a set of mask paths for O(1) lookups
        mask_set = set(self.masks_list)
        
        self.valid_images = []
        self.valid_masks = []
        
        for img_name in self.images:
            # Reconstruct mask filename from image, based on convert.py's naming logic
            # volume-XX_slice_YYYY.png -> segmentation-XX_slice_YYYY.png
            mask_name = img_name.replace('volume-', 'segmentation-')
            if mask_name in mask_set:
                self.valid_images.append(img_name)
                self.valid_masks.append(mask_name)

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.valid_images[idx])
        mask_path = os.path.join(self.masks_dir, self.valid_masks[idx])
        
        # Load grayscale images
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        
        # Convert to numpy and normalize image to [0, 1]
        image = np.array(image, dtype=np.float32) / 255.0
        
        # Process mask mapping: 0 -> 0 (bg), 127 -> 1 (liver), 255 -> 2 (tumor)
        mask_array = np.array(mask, dtype=np.int64)
        mask_processed = np.zeros_like(mask_array, dtype=np.int64)
        
        # Using thresholds to account for any slight saving noise, though PNG should be exact
        mask_processed[mask_array >= 100] = 1 # Liver (was ~127)
        mask_processed[mask_array >= 200] = 2 # Tumor (was ~255)
        
        # Image shape: (1, H, W)
        image = torch.from_numpy(image).unsqueeze(0)
        
        # Mask shape: (H, W)
        mask_processed = torch.from_numpy(mask_processed)
        
        return image, mask_processed