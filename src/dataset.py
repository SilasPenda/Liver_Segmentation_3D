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
from utils import get_config


class LoadTransformDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.images = sorted(os.listdir(images_dir))
        self.labels = sorted(os.listdir(labels_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
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

    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,  num_workers=2)

    return train_loader, val_loader



if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders()

    for image, label in val_loader:
        slice_idx = image.shape[2] // 2  # Take the middle slice along the depth axis
        print(np.unique(label[0]))
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
