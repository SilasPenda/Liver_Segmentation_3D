import os
import argparse
import torch.amp
from tqdm import tqdm

import torch
import torch.nn as nn

from utils.dataset import get_data_loaders

from utils.models import UNet2D
from utils.U_Net_utils import get_config, dice_score, save_checkpoint, load_checkpoint

def main():
    parser = argparse.ArgumentParser(description='Train a 2D U-Net Segmentation model.')
    parser.add_argument('-e', '--epochs', type=int, help='Number of Epochs', required=True)
    parser.add_argument('-p', '--checkpoint', type=str, default=None, help='Model checkpoint', required=False)
    args = parser.parse_args()

    # Loading config and setting up paths (same as before)
    config = get_config(config_filepath='./config.yaml')
    n_classes = config.get('n_classes', 1)
    img_channels = config.get('img_channels', 1)

    # if n_classes == 1:
    #     loss_fn = nn.BCEWithLogitsLoss()
    #     activ_func = "Sigmoid"
    # else:
    criterion = nn.CrossEntropyLoss()

    # scaler = torch.GradScaler()

    IMAGE_HEIGHT = 160
    IMAGE_WIDTH = 240

    train_loader, val_loader = get_data_loaders(IMAGE_HEIGHT, IMAGE_WIDTH)
    
    num_epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UNet2D(img_channels, n_classes)
    model.to(device)

    if args.checkpoint is not None:
        load_checkpoint(torch.load(args.checkpoint), model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_loss = 0  # Initialize best validation Dice score

    train_loss = 0
    val_loss = 0

    os.makedirs("models", exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = 0  # Reset at the start of each epoch
        val_loss = 0

        # Training phase
        model.train()
        for images, masks in train_loader:
          
          images = images.to(device, dtype=torch.float32)
          masks = masks.to(device, dtype=torch.long).squeeze(1)
      
          # Forward pass
          pred = model(images)
          loss = criterion(pred, masks)
          train_loss += loss.item()
      
          # dice = dice_score(pred , masks, n_classes)

          # Backward pass
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
          for images, masks in val_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long).squeeze(1)

            pred = model(images)
            loss = criterion(pred, masks)
            val_loss += loss.item()
            
            # Calculate Dice score
            # dice = dice_score(pred, masks, n_classes)
            # val_dice += dice.item()

        # Compute mean losses and Dice scores
        avg_train_loss = train_loss / len(train_loader)
        # avg_train_dice = train_dice / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        # avg_val_dice = val_dice / len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.2f}, - "
            f"Val Loss: {avg_val_loss:.2f}")

        # Save the best model based on validation Dice score
        if avg_val_loss < best_loss:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=os.path.join("models", "best_model.pth.tar"))
            best_loss = avg_val_loss
            print(f'Best model updated at epoch {epoch + 1}')

        # Save the latest model after each epoch
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=os.path.join("models", "last_model.pth.tar"))
                        

if __name__ == "__main__":
    main()