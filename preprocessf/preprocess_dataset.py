import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import torch.nn as nn
from unet import UNet
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(os.path.join(root_dir, "images")) if f.endswith(".jpg")]
    
    def __len__(self):
        return len(self.image_files)
    
    @staticmethod
    def preprocess(pil_img_arr,scale):
        pil_img=pil_img_arr[0]    # extract image from array
        
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        

        pil_img_label=pil_img_arr[1]   #extract label from array

        if pil_img_label=='mask':   
            mask_values=[0,1]                
            pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST)
            img = np.asarray(pil_img)
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask
        else:   
            pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
            img = np.asarray(pil_img)             
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0
            return img
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.root_dir, "images", image_name)
        mask_path = os.path.join(self.root_dir, "masks", image_name.replace(".jpg", "_mask.gif"))  # Example mask naming
        
        image = [Image.open(image_path),'image'] #read image and label as image
        # image= np.asarray(image)
        mask =[Image.open(mask_path),'mask']  #read mask and label as mask
        # mask= np.asarray(image)
        
        scale=0.3
        image= self.preprocess(image,scale)
        mask= self.preprocess(mask,scale)
               
        
        return torch.as_tensor(image.copy()).float().contiguous(), torch.as_tensor(mask.copy()).float().contiguous()
        
   
            
