
from unet import UNet
from os import listdir
import torch
import argparse
import logging
from os.path import splitext, isfile, join
import os
from preprocessf import CustomDataset
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()





# def get_args():
#     parser = argparse.ArgumentParser(description='Predict masks from input images')
#     parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
#                         help='Specify the file in which the model is stored')
#     parser.add_argument('--input', '-i',help='Path to the root directory', required=True)
#     parser.add_argument('--output', '-o', help='Path to the output directory', required=True)
#     parser.add_argument('--viz', '-v', action='store_true',
#                         help='Visualize the images as they are processed')
#     parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
#     parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
#                         help='Minimum probability value to consider a mask pixel white')
#     parser.add_argument('--scale', '-s', type=float, default=0.5,
#                         help='Scale factor for the input images')
#     parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
#     parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
#     return parser.parse_args()

if __name__ == '__main__':
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    #args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')   
    image_path='C://RICARDO//2023 CISTECH BACKUP//cistech//2023//NRCan//Carvana 1//Unet-Carvana//Dataset//Test//Images//c7a94c46a3b2_05.jpg'
    full_img=Image.open(image_path)
    image_arr = [full_img,'image'] #read image and label as image
    img=torch.from_numpy(CustomDataset.preprocess(image_arr,0.3))
    img = img.to(device=device, dtype=torch.float32)

    model = UNet(n_channels=3, n_classes=1, bilinear=True)  #n_classes is one because we are trying to detect segmentation.
    model.to(device=device)
    # Load the saved state dictionary
    load_path = "model_checkpoint.pth"
    model.load_state_dict(torch.load(load_path))
    model.eval()  # Set the model to evaluation mode
    print(model)
    out_threshold=0.5
   
    img=img.unsqueeze(0)       

    with torch.no_grad():
        output=model(img)
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if model.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold
    
    mask=mask[0].long().squeeze().numpy()
    print(mask)
    plot_img_and_mask(full_img, mask)
            

            
        


  
    

    



