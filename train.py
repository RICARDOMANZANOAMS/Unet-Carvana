import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
import torch.nn as nn
from unet import UNet
import numpy as np

torch.cuda.empty_cache()

# # Create a U-Net model
# in_channels = 3  # RGB channels
# out_channels = 1  # Single channel mask
# model = UNet(in_channels, out_channels)

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(os.path.join(root_dir, "images")) if f.endswith(".jpg")]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.root_dir, "images", image_name)
        mask_path = os.path.join(self.root_dir, "masks", image_name.replace(".jpg", "_mask.gif"))  # Example mask naming
        
        image = Image.open(image_path)
        # image= np.asarray(image)
        mask = Image.open(mask_path)
        # mask= np.asarray(image)
        @staticmethod
        def preprocess_mask(pil_img,scale):
            mask_values=[0,1]
            w, h = pil_img.size
            newW, newH = int(scale * w), int(scale * h)
            pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST)
            img = np.asarray(pil_img)

           
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask
        
        @staticmethod
        def preprocess_image(pil_img,scale):
            
            w, h = pil_img.size
            newW, newH = int(scale * w), int(scale * h)
            
            pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
            img = np.asarray(pil_img)

            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0
            return img
        scale=0.2
        image= preprocess_image(image,scale)
        mask= preprocess_mask(mask,scale)
               

        return torch.as_tensor(image.copy()).float().contiguous(), torch.as_tensor(mask.copy()).float().contiguous()
    
def train_model(model):
    batch_size = 2
    image_size = 256
   
    train_dataset = CustomDataset(root_dir="C:/RICARDO/2023 CISTECH BACKUP/cistech/2023/NRCan/Carvana 1/Unet-Carvana/Dataset/short_dataset/train")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Similarly, create a validation dataset and loader



    criterion =  nn.BCEWithLogitsLoss()# Binary Cross-Entropy loss for binary segmentation
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
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
    model = UNet( 1, 3)  #n_classes is one because we are trying to detect segmentation.
    model.to(device) 
    #try:
    model=train_model(model)
   
           