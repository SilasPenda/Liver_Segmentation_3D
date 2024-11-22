import os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from utils.U_Net_utils import get_config


class LoadTransformDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.images = os.listdir(images_dir)
        self.transform = transform

        self.color_to_class = {
            (0, 0, 0): 0,       # Background
            (255, 0, 0): 1,     # elongated
            (0, 255, 0): 2,     # circular
            (0, 0, 255): 3      # other
        }

    def __len__(self):
        return len(self.images)

    def rgb_to_class(self, mask):
      """Convert an RGB mask to class indices."""
      class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
      for rgb, cls in self.color_to_class.items():
          class_mask[np.all(mask == rgb, axis=-1)] = cls
          
      return class_mask
    
    def __getitem__(self, idx):
        # Get image and corresponding label paths
        image_path = os.path.join(self.images_dir, self.images[idx])
        rgb_mask_path = os.path.join(self.labels_dir, f"mask_{self.images[idx].split('.')[0].split('_')[1]}.png")
        
        # Open image and mask
        image = np.array(Image.open(image_path))
        rgb_mask = np.array(Image.open(rgb_mask_path))

        mask = self.rgb_to_class(rgb_mask)

        # Ensure mask is single-channel (convert if necessary)
        # if len(mask.shape) == 3:  # If the mask is RGB
        #     mask = mask[:, :, 0]

        # image = image.transpose((2, 0, 1))
        # mask = mask.transpose((2, 0, 1))

      # # For each channel in the mask, set pixels to 1.0 where the pixel value is 255
      # for i in range(3):  # Loop over the 3 channels
      #     mask[:, :, i] = np.where(mask[:, :, i] == 255, 1.0, 0.0)  # Convert white pixels (255) to 1.0, else 0.0

      # Convert both image and mask to torch tensors (assuming you want to use these for PyTorch)
      # image = torch.tensor(image, dtype=torch.float32)
      

        # Apply transformations, if any
        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = image.type(torch.FloatTensor)
        mask = mask.type(torch.FloatTensor)
        
        return image, mask


    
def pad_tensor(tensor, max_height, max_width):
    # Get the current shape of the tensor
    _, height, width = tensor.shape  # First dimension is the channel

    # Calculate the padding required
    pad_height = max_height - height
    pad_width = max_width - width

    # Pad the tensor (pad only height and width, not channels)
    padded_tensor = F.pad(tensor, (0, pad_width, 0, pad_height))  # (left, right, top, bottom)
    
    return padded_tensor

def pad_collate_fn(batch):
    # Get the maximum height and width for the batch
    max_height = max([item[0].shape[-2] for item in batch])  # item[0] is the image
    max_width = max([item[0].shape[-1] for item in batch])

    # Apply padding to each image and label
    padded_images = [pad_tensor(img, max_height, max_width) for img, _ in batch]
    padded_masks = [pad_tensor(mask, max_height, max_width) for _, mask in batch]

    # Stack the images and labels into batches
    batch_images = torch.stack(padded_images)
    batch_masks = torch.stack(padded_masks)

    return batch_images, batch_masks



def get_data_loaders(IMAGE_HEIGHT, IMAGE_WIDTH):
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # Loading config and setting up paths (same as before)
    config = get_config(config_filepath='./config.yaml')
    train_path = config.get('train_path', None)
    val_path = config.get('val_path', None)

    train_images_dir = os.path.join(train_path, "images")
    train_labels_dir = os.path.join(train_path, "masks")
    val_images_dir = os.path.join(val_path, "images")
    val_labels_dir = os.path.join(val_path, "masks")

    # Create dataset and DataLoader
    train_dataset = LoadTransformDataset(images_dir=train_images_dir, labels_dir=train_labels_dir, transform=train_transform)
    val_dataset = LoadTransformDataset(images_dir=val_images_dir, labels_dir=val_labels_dir, transform=val_transforms)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    return train_loader, val_loader