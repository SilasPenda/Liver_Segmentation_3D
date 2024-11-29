import os
import numpy as np
# from skimage import io
import nibabel as nib
import matplotlib.pyplot as plt
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from src.utils import get_config


class LoadTransformDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.images = sorted(os.listdir(images_dir))
        self.labels = sorted(os.listdir(labels_dir))
        self.transform = transform

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
        label_path = os.path.join(self.labels_dir, self.labels[idx])

        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        image = np.transpose(image, (2, 1, 0))
        label = np.transpose(label, (2, 1, 0))

        # Convert both image and label to torch tensors
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        image = image.unsqueeze(0)
        label = label.unsqueeze(0)
        # print(image.shape)
        # print(label.shape)
        # Apply transformations, if any, to both the image and the label
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

    
def pad_tensor(tensor, max_depth, max_height, max_width):
    # Get the current shape of the tensor
    depth, height, width = tensor.shape[-3], tensor.shape[-2], tensor.shape[-1]

    # Calculate the padding required
    pad_depth = max_depth - depth
    pad_height = max_height - height
    pad_width = max_width - width

    # Pad the tensor
    padded_tensor = F.pad(tensor, (0, pad_width, 0, pad_height, 0, pad_depth))
    
    return padded_tensor

def pad_collate_fn(batch):
    # Get the maximum depth, height, and width for the batch
    max_depth = max([item[0].shape[-3] for item in batch]) 
    max_height = max([item[0].shape[-2] for item in batch])
    max_width = max([item[0].shape[-1] for item in batch])

    # Apply padding to each image and label
    padded_images = [pad_tensor(img, max_depth, max_height, max_width) for img, _ in batch]
    padded_labels = [pad_tensor(lbl, max_depth, max_height, max_width) for _, lbl in batch]

    # Stack the images and labels into batches
    batch_images = torch.stack(padded_images)
    batch_labels = torch.stack(padded_labels)

    return batch_images, batch_labels

def get_data_loaders():
    # Loading config and setting up paths (same as before)
    config = get_config(config_filepath=os.path.join(os.getcwd(), "config.yaml"))
    # print(os.path.join(os.getcwd(), "config.yaml"))
    train_path = config.get("train_path", None)
    val_path = config.get("val_path", None)

    train_images_dir = os.path.join(train_path, "images")
    train_labels_dir = os.path.join(train_path, "labels")
    val_images_dir = os.path.join(val_path, "images")
    val_labels_dir = os.path.join(val_path, "labels")

    # Create dataset and DataLoader
    train_dataset = LoadTransformDataset(images_dir=train_images_dir, labels_dir=train_labels_dir)
    val_dataset = LoadTransformDataset(images_dir=val_images_dir, labels_dir=val_labels_dir)

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=2)

    return train_loader, val_loader



# def get_data_loaders(IMAGE_HEIGHT, IMAGE_WIDTH):
    # train_transform = A.Compose(
    #     [
    #         # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    #         A.Rotate(limit=35, p=1.0),
    #         A.HorizontalFlip(p=0.5),
    #         A.VerticalFlip(p=0.1),
    #         A.Normalize(
    #             mean=[0.0, 0.0, 0.0],
    #             std=[1.0, 1.0, 1.0],
    #             max_pixel_value=255.0,
    #         ),
    #         ToTensorV2(),
    #     ],
    # )

    # val_transforms = A.Compose(
    #     [
    #         # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    #         A.Normalize(
    #             mean=[0.0, 0.0, 0.0],
    #             std=[1.0, 1.0, 1.0],
    #             max_pixel_value=255.0,
    #         ),
    #         ToTensorV2(),
    #     ],
    # )

    # # Loading config and setting up paths (same as before)
    # config = get_config(config_filepath='./config.yaml')
    # train_path = config.get('train_path', None)
    # val_path = config.get('val_path', None)

    # train_images_dir = os.path.join(train_path, "images")
    # train_labels_dir = os.path.join(train_path, "masks")
    # val_images_dir = os.path.join(val_path, "images")
    # val_labels_dir = os.path.join(val_path, "masks")

    # # Create dataset and DataLoader
    # train_dataset = LoadTransformDataset(images_dir=train_images_dir, labels_dir=train_labels_dir, transform=train_transform)
    # val_dataset = LoadTransformDataset(images_dir=val_images_dir, labels_dir=val_labels_dir, transform=val_transforms)

    # batch_size = 2
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # return train_loader, val_loader



if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders()

    for image, label in val_loader:
        slice_idx = image.shape[2] // 2  # Take the middle slice along the depth axis

        plt.figure(figsize=(10, 10))

        # Display the middle slice of the image
        plt.subplot(2, 2, 1)
        plt.imshow(image[0, 0, slice_idx, :, :], cmap="gray")  # Assuming (batch, channels, depth, height, width)
        plt.title("Image Slice")
        plt.axis("off")

        # Display the middle slice of the label
        plt.subplot(2, 2, 2)
        plt.imshow(label[0, 0, slice_idx, :, :], cmap="gray")  # Assuming the label has the same shape
        plt.title("Label Slice")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        break
