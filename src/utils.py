import yaml
import torch

def save_checkpoint(state, filename="my_checkpoint.pth"):
    # print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_config(config_filepath: str) -> dict:
    try:
        with open(config_filepath) as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        return {}
    
# def dice_loss(pred, target, smooth=1):
#     # pred: [batch, channels, depth, height, width], softmax probabilities
#     # target: [batch, depth, height, width], indices of classes (0 to C-1)

#     # Apply softmax to obtain probabilities
#     pred = torch.softmax(pred, dim=1)
#     C = pred.shape[1]  # Number of classes

#     dice = 0
#     for c in range(C):
#         pred_c = pred[:, c, :, :, :]  # Get probabilities for class c
#         target_c = (target == c).float()  # Create a mask for class c

#         # Compute intersection and union over 3D spatial dimensions
#         intersection = (pred_c * target_c).sum((1, 2, 3))  # Sum over [depth, height, width]
#         union = pred_c.sum((1, 2, 3)) + target_c.sum((1, 2, 3))

#         dice_c = (2.0 * intersection + smooth) / (union + smooth)
#         dice += dice_c.mean()  # Average over all images in the batch

#     return 1.0 - dice / C  # Average over all classes

    
    
# def dice_coefficient_loss(pred, target, smooth=1e-6):
#     # pred: [batch, channels, height, width], pred should be softmax probabilities
#     # target: [batch, height, width], target should be indices of classes (0 to C-1)

#     # print(f"pred shape: {pred.shape}")
#     # print(f"target shape: {target.shape}")

    
#     pred = torch.softmax(pred, dim=1)  # Apply softmax to obtain probabilities
#     C = pred.shape[1]  # Number of classes

#     dice = 0
#     for c in range(C):
#         pred_c = pred[:, c, :, :]
#         target_c = (target == c).float()  # Create a mask for class c

#         intersection = (pred_c * target_c).sum((1, 2))  # Sum over each image separately
#         union = pred_c.sum((1, 2)) + target_c.sum((1, 2))

#         dice_c = (2. * intersection + smooth) / (union + smooth)
#         dice += dice_c.mean()  # Average over all images in the batch

#     return 1.0 - dice / C  # Average over all classes



def dice_coefficient_loss(pred, target, smooth=1):
    pred = torch.softmax(pred, dim=1)  # Assuming multi-class segmentation
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.size(1)).permute(0, 4, 1, 2, 3).float()
    # print(target_one_hot.shape)

    intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - dice.mean()


    
# def dice_score(pred, target, smooth=1e-6):
#     # Convert prediction to binary using a threshold (0.5 for binary segmentation)
#     pred = (pred > 0.5).float()
    
#     # Flatten the tensors
#     preds = pred.view(-1)
#     targets = target.view(-1)

#     # Calculate intersection and union
#     intersection = (preds * targets).sum()
#     union = preds.sum() + targets.sum()

#     # Calculate Dice coefficient
#     dice = (2. * intersection + smooth) / (union + smooth)

#     return dice

# def dice_score(pred, target, num_classes):
#     """Calculate the Dice score for multiclass segmentation."""
#     smooth = 1e-6
#     dice = 0.0

#     # Apply softmax to get class probabilities and then calculate Dice per class
#     for i in range(num_classes):
#         # Binary mask for class i
#         pred_i = pred[:, i, :, :]  # Predicted probabilities for class i
#         target_i = (target == i).float()  # True mask for class i

#         # Calculate Dice for class i
#         intersection = (pred_i * target_i).sum()
#         union = pred_i.sum() + target_i.sum()
#         dice += (2.0 * intersection + smooth) / (union + smooth)

#     # Average Dice score across all classes
#     return dice / num_classes

