
import cv2
from PIL import Image
import numpy as np
img_path='C:/RICARDO/2023 CISTECH BACKUP/cistech/2023/NRCan/Carvana 1/Unet-Carvana/Dataset/test_dataset/images/0cdf5b5d0ce1_01.jpg'
mask_path='C:/RICARDO/2023 CISTECH BACKUP/cistech/2023/NRCan/Carvana 1/Unet-Carvana/Dataset/test_dataset/masks/0cdf5b5d0ce1_01_mask.gif'
img = cv2.imread(img_path)
print(img.shape)
print(type(img))
print(img)

print("NUMPY MASK")
mask = Image.open(mask_path)    #Object of PIL 
#print(mask)

print(type(mask))
numpy_array = np.asarray(mask)
print(numpy_array.shape)


img = Image.open(img_path)    #Object of PIL 
print("NUMPY IMG")

print(type(img ))
img_numpy_array = np.asarray(img )
print(img_numpy_array.shape)