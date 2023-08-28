
from unet import UNet
import torch

model = UNet(n_channels=3, n_classes=1, bilinear=True)  #n_classes is one because we are trying to detect segmentation.

# Load the saved state dictionary
load_path = "model_checkpoint.pth"
model.load_state_dict(torch.load(load_path))
model.eval()  # Set the model to evaluation mode
