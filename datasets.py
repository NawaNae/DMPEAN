import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets
from PIL import Image, ImageFile
from skimage import io, transform
import os
from os import listdir
from os.path import join
import numpy as np
import random
import re
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SmartDocQADataset(Dataset):
    def __init__(self, blur_image_pathes, sharp_image_root, Center_Crop = False, Random_Crop=False,center_crop_size = (1024,1024) ,random_crop_size=256, multi_scale=False, rotation=False, color_augment=False, transform=None):
        self.blur_image_files = []
        for path in blur_image_pathes:
            files = [f"{path}{f}" for f in listdir(path)]
            self.blur_image_files += files
        self.transform = transform
        self.sharp_image_root = sharp_image_root
        self.Center_Crop = Center_Crop
        self.Random_Crop = Random_Crop
        self.center_crop_size = center_crop_size
        self.random_crop_size = random_crop_size
    def __len__(self):
        return len(self.blur_image_files)

    def __getitem__(self, index):
        match = "^[SM]_Img_.*_D\d{1,2}_L\d{1,2}_r35_a-{0,1}\d{1,2}_b-{0,1}\d{1,2}"
        blurimgP = self.blur_image_files[index]
        bpsp = blurimgP.split("/")
        blurimgName = bpsp[-1]
        phone = bpsp[-3]
        try:
            ans = re.search(match, blurimgName).group()
        except:
            pass

        sharpimgP = f"{self.sharp_image_root}{phone}/Images/{ans}.jpg"
        blurimg = Image.open(blurimgP).convert('RGB')
        sharpimg = Image.open(sharpimgP).convert('RGB')
        # blurimg.save('original_blur_img.jpg')
        # sharpimg.save('original_sharp_img.jpg')
        # im1 = blurimg.save("blurimg.jpg")
        # im1 = im1
        if self.transform:
            blurimg = self.transform(blurimg)
            sharpimg = self.transform(sharpimg)

        if self.Center_Crop:
            W = blurimg.size()[1]
            H = blurimg.size()[2]
            W_crop = self.center_crop_size[0] // 2
            H_crop = self.center_crop_size[1] // 2
            try:
                blurimg = blurimg[:, W // 2 - W_crop : W//2 + W_crop, H//2 - H_crop:H//2 + H_crop]
            except:
                pass
            sharpimg = sharpimg[:, W // 2 - W_crop : W//2 + W_crop, H//2 - H_crop:H//2 + H_crop]
        PreviewPath="./preprocessPreview/"
        
        # torchvision.utils.save_image(blurimg,f"{PreviewPath}{blurimgName}_blur.jpg")
        # torchvision.utils.save_image(sharpimg,f"{PreviewPath}{blurimgName}_sharp.jpg")
        if self.Random_Crop:
            W = blurimg.size()[1]
            H = blurimg.size()[2]

            Ws = np.random.randint(0, W-self.random_crop_size-1, 1)[0]
            Hs = np.random.randint(0, H-self.random_crop_size-1, 1)[0]

            blurimg = blurimg[:, Ws:Ws +
                                    self.random_crop_size, Hs:Hs+self.random_crop_size]
            sharpimg = sharpimg[:, Ws:Ws +
                                    self.random_crop_size, Hs:Hs+self.random_crop_size]

        return {'blur_image': blurimg, 'sharp_image': sharpimg}


class GoProDataset(Dataset):
    def __init__(self, blur_image_files, sharp_image_files, root_dir, crop=False, crop_size=256, multi_scale=False, rotation=False, color_augment=False, transform=None):
        """
        Args:
             split_file: Path to the split file
             root_dir: Directory with all the images
             transform: Optional transform to be appeared on a sample
        """
        blur_file = open(blur_image_files, 'r')
        self.blur_image_files = blur_file.readlines()
        sharp_file = open(sharp_image_files, 'r')
        self.sharp_image_files = sharp_file.readlines()
        self.root_dir = root_dir
        self.transform = transform
        self.crop = crop
        self.crop_size = crop_size
        self.multi_scale = multi_scale
        self.rotation = rotation
        self.color_augment = color_augment
        self.rotate90 = transforms.RandomRotation(90)
        self.rotate45 = transforms.RandomRotation(45)

    def __len__(self):
        return len(self.blur_image_files)

    def __getitem__(self, idx):
        image_name = self.blur_image_files[idx][0:-1].split('/')
        blur_image = Image.open(os.path.join(
            self.root_dir, image_name[0], image_name[1], image_name[2], image_name[3])).convert('RGB')
        sharp_image = Image.open(os.path.join(
            self.root_dir, image_name[0], image_name[1], 'sharp', image_name[3])).convert('RGB')

        if self.rotation:
            degree = random.choice([90, 180, 270])
            blur_image = transforms.functional.rotate(blur_image, degree)
            sharp_image = transforms.functional.rotate(sharp_image, degree)

        if self.color_augment:
            #contrast_factor = 1 + (0.2 - 0.4*np.random.rand())
            #blur_image = transforms.functional.adjust_contrast(blur_image, contrast_factor)
            #sharp_image = transforms.functional.adjust_contrast(sharp_image, contrast_factor)
            blur_image = transforms.functional.adjust_gamma(blur_image, 1)
            sharp_image = transforms.functional.adjust_gamma(sharp_image, 1)
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            blur_image = transforms.functional.adjust_saturation(
                blur_image, sat_factor)
            sharp_image = transforms.functional.adjust_saturation(
                sharp_image, sat_factor)

        if self.transform:
            blur_image = self.transform(blur_image)
            sharp_image = self.transform(sharp_image)

        if self.crop:
            W = blur_image.size()[1]
            H = blur_image.size()[2]

            Ws = np.random.randint(0, W-self.crop_size-1, 1)[0]
            Hs = np.random.randint(0, H-self.crop_size-1, 1)[0]

            blur_image = blur_image[:, Ws:Ws +
                                    self.crop_size, Hs:Hs+self.crop_size]
            sharp_image = sharp_image[:, Ws:Ws +
                                      self.crop_size, Hs:Hs+self.crop_size]

        if self.multi_scale:
            H = sharp_image.size()[1]
            W = sharp_image.size()[2]
            blur_image_s1 = transforms.ToPILImage()(blur_image)
            sharp_image_s1 = transforms.ToPILImage()(sharp_image)
            blur_image_s2 = transforms.ToTensor()(
                transforms.Resize([H/2, W/2])(blur_image_s1))
            sharp_image_s2 = transforms.ToTensor()(
                transforms.Resize([H/2, W/2])(sharp_image_s1))
            blur_image_s3 = transforms.ToTensor()(
                transforms.Resize([H/4, W/4])(blur_image_s1))
            sharp_image_s3 = transforms.ToTensor()(
                transforms.Resize([H/4, W/4])(sharp_image_s1))
            blur_image_s1 = transforms.ToTensor()(blur_image_s1)
            sharp_image_s1 = transforms.ToTensor()(sharp_image_s1)
            return {'blur_image_s1': blur_image_s1, 'blur_image_s2': blur_image_s2, 'blur_image_s3': blur_image_s3, 'sharp_image_s1': sharp_image_s1, 'sharp_image_s2': sharp_image_s2, 'sharp_image_s3': sharp_image_s3}
        else:
            return {'blur_image': blur_image, 'sharp_image': sharp_image}
