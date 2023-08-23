import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    # a staticmethod does not belong an instance. It is independent

    @staticmethod
    #We pass an Image or a Mask. NOT BOTH at the same time
    def preprocess(mask_values, pil_img, scale, is_mask): #this function takes pil_img( It can be an image or mask)

        w, h = pil_img.size   #get the size of the image (width, heigth)
        newW, newH = int(scale * w), int(scale * h)  #scale the image depending on the value of scale and round with int
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'  #verify scale image
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)  #resize the image depending on the type of image
        img = np.asarray(pil_img)  # we transform the image to array
        #if it is a color image, it has 3 arrays inside it
        #if it is a greyscale, we get only 1 array


        #if the image provided is a mask, it handles it 
        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:  #verify if the mask image is two dimensional or three dimensional
                    mask[img == v] = i   #case gray scale
                else:    #case color mask
                    mask[(img == v).all(-1)] = i

            return mask
        #if the image provided is an image not a mask
        else:
            if img.ndim == 2:   #if it is grayscale
                img = img[np.newaxis, ...]  #if it is grayscale, we increase one dimension at the beginning
            else:
                img = img.transpose((2, 0, 1)) #if it is color image, we move the third the dimension to the beginning
                                               # channel dimension (usually the third dimension) comes first, followed by the height and width dimensions

            if (img > 1).any():       # we normalize the image
                img = img / 255.0

            return img

    #This function is called every time that we iterate through an array of elements. To illustrate, when we call array[0], function getitem
    #will be executed. In this case, we do not called explicity. However, when the dataloader is called, the getitem function is called
    #This function access to each datasample in an array; thus, we can process any image here in this function
    def __getitem__(self, idx):  
        name = self.ids[idx]    #get the name of the image
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))  #get the mask_file of the image with name "name"
        img_file = list(self.images_dir.glob(name + '.*'))     #get the image with name "name"

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}' #verify if we have two images with same name
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}' #verify if we have two masks with same name
        #In this case, the mask has only two dimensions white and black. Thus, we obtain only one array
        mask = load_image(mask_file[0])  #Load mask. It uses Image.open(filename) since we have .jpg
        #In this case, we have a color image; thus, we get a three vector array 
        #We get one dimension for red, another for blue, and another for gray  RGB
        img = load_image(img_file[0])    #Load image. It uses Image.open(filename) since we have .jpg

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'  #verify size of mask and img should be the same size

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)   #Execute the preprocess function which is a static method
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True) #Execute the preprocess function which is a static method

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
