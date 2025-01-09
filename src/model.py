import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet3D, self).__init__()
        
        # Define layers for the encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Define layers for the decoder
        self.upconv4 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Final output layer
        self.conv_out = nn.Conv3d(64, num_classes, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool3d(enc1, 2))
        enc3 = self.enc3(F.max_pool3d(enc2, 2))
        enc4 = self.enc4(F.max_pool3d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool3d(enc4, 2))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = self._resize_and_concat(dec4, enc4)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self._resize_and_concat(dec3, enc3)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self._resize_and_concat(dec2, enc2)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self._resize_and_concat(dec1, enc1)
        dec1 = self.dec1(dec1)

        return self.conv_out(dec1)

    def _resize_and_concat(self, x, skip):
        """Resize x to match skip's spatial dimensions and concatenate."""
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)
        return torch.cat([x, skip], dim=1)


def test():
    input_channels = 1
    n_classes = 3

    x = torch.randn((1, input_channels, 54, 161, 161))  # (batch_size, channels, depth, height, width)
    model = UNet3D(input_channels, n_classes)
    preds = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Predicted shape: {preds.shape}")

    print(summary(model, input_size=(input_channels, 54, 161, 161)))



if __name__ == "__main__":
    test()