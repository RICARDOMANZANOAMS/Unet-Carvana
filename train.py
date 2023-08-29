import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
import torch.nn as nn
from unet import UNet
import numpy as np
from preprocessf import CustomDataset
torch.cuda.empty_cache()

# # Create a U-Net model
# in_channels = 3  # RGB channels
# out_channels = 1  # Single channel mask
# model = UNet(in_channels, out_channels)

    
def train_model(model):
    batch_size = 2
    image_size = 256
   
    train_dataset = CustomDataset(root_dir="C:/RICARDO/2023 CISTECH BACKUP/cistech/2023/NRCan/Carvana 1/Unet-Carvana/Dataset/short_dataset/train")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Similarly, create a validation dataset and loader



    criterion =  nn.BCEWithLogitsLoss()# Binary Cross-Entropy loss for binary segmentation
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        print("epoch")
        print(epoch)
        for images, masks in train_loader:
            # print("Length batch")
            # print(len(images))
            # print("images")
            # print(images)
            # print("masks")
            # print(masks)
            # print(masks.shape)
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            masks_pred = model(images)
            # print("before eliminating one dimension")
            # print(masks_pred.shape)
            # print(masks_pred)
            masks_pred =masks_pred.squeeze(1)
            # print("after eliminating one dimension")
            # print(masks_pred.shape)
            # print(masks_pred)
            loss = criterion(masks_pred, masks.float())
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    save_path = "model_checkpoint.pth"
    torch.save(model.state_dict(), save_path)
    return model
    
if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=1, bilinear=True)  #n_classes is one because we are trying to detect segmentation.
    model.to(device) 
    print(model)
    #try:
    model=train_model(model)
   
           