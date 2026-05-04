# import os
# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# import numpy as np

# class LiTSDataset(Dataset):
#     def __init__(self, images_dir, masks_dir):
#         self.images_dir = images_dir
#         self.masks_dir = masks_dir
        
#         # Look for PNG files
#         self.images = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
#         self.masks_list = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])

#         # Create a set of mask paths for O(1) lookups
#         mask_set = set(self.masks_list)
        
#         self.valid_images = []
#         self.valid_masks = []
        
#         for img_name in self.images:
#             # Reconstruct mask filename from image, based on convert.py's naming logic
#             # volume-XX_slice_YYYY.png -> segmentation-XX_slice_YYYY.png
#             mask_name = img_name.replace('volume-', 'segmentation-')
#             if mask_name in mask_set:
#                 self.valid_images.append(img_name)
#                 self.valid_masks.append(mask_name)

#     def __len__(self):
#         return len(self.valid_images)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.images_dir, self.valid_images[idx])
#         mask_path = os.path.join(self.masks_dir, self.valid_masks[idx])
        
#         # Load grayscale images
#         image = Image.open(img_path).convert("L")
#         mask = Image.open(mask_path).convert("L")
        
#         # Convert to numpy and normalize image to [0, 1]
#         image = np.array(image, dtype=np.float32) / 255.0
        
#         # Process mask mapping: 0 -> 0 (bg), 127 -> 1 (liver), 255 -> 2 (tumor)
#         mask_array = np.array(mask, dtype=np.int64)
#         mask_processed = np.zeros_like(mask_array, dtype=np.int64)
        
#         # Using thresholds to account for any slight saving noise, though PNG should be exact
#         mask_processed[mask_array >= 100] = 1 # Liver (was ~127)
#         mask_processed[mask_array >= 200] = 2 # Tumor (was ~255)
        
#         # Image shape: (1, H, W)
#         image = torch.from_numpy(image).unsqueeze(0)
        
#         # Mask shape: (H, W)
#         mask_processed = torch.from_numpy(mask_processed)
        
#         return image, mask_processed

import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from scipy.ndimage import map_coordinates, gaussian_filter


class LiTSDataset(Dataset):

    def __init__(self, images_dir, masks_dir, augment=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment

        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
        self.masks = sorted([f for f in os.listdir(masks_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.images)

    # ---------------------------------------------------
    # CLAHE
    # ---------------------------------------------------
    def clahe(self, img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(img)

    # ---------------------------------------------------
    # Elastic Transform
    # ---------------------------------------------------
    def elastic(self, image, mask, alpha=20, sigma=3):

        random_state = np.random.RandomState(None)
        shape = image.shape

        dx = gaussian_filter((random_state.rand(*shape)*2-1), sigma)*alpha
        dy = gaussian_filter((random_state.rand(*shape)*2-1), sigma)*alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

        indices = np.reshape(y+dy, (-1,1)), np.reshape(x+dx, (-1,1))

        image = map_coordinates(image, indices, order=1).reshape(shape)
        mask = map_coordinates(mask, indices, order=0).reshape(shape)

        return image, mask

    # ---------------------------------------------------
    # ROI Crop
    # ---------------------------------------------------
    def roi_crop(self, image, mask):

        coords = np.argwhere(mask > 0)

        if len(coords) == 0:
            return image, mask

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)

        image = image[y0:y1+1, x0:x1+1]
        mask = mask[y0:y1+1, x0:x1+1]

        image = cv2.resize(image, (256,256))
        mask = cv2.resize(mask, (256,256), interpolation=cv2.INTER_NEAREST)

        return image, mask

    def __getitem__(self, idx):

        img = Image.open(os.path.join(self.images_dir, self.images[idx])).convert("L")
        mask = Image.open(os.path.join(self.masks_dir, self.masks[idx])).convert("L")

        img = np.array(img, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)

        # CLAHE
        img = self.clahe(img)

        # Mask convert
        m = np.zeros_like(mask)
        m[mask >= 100] = 1
        m[mask >= 200] = 2

        # ROI
        img, m = self.roi_crop(img, m)

        # Elastic
        if self.augment:
            img, m = self.elastic(img, m)

        img = img.astype(np.float32) / 255.0

        img = torch.tensor(img).unsqueeze(0).float()
        m = torch.tensor(m).long()

        return img, m