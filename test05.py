import torch
from datasets import SmartDocQADataset
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets
# Dataset
DATASET_PATH = "./dataset/smartDocQA/"
PHONES = ["Nokia_phone", "Samsung_phone"]
BLUR_PATH = f"{DATASET_PATH}Captured_Images/"
SHARP_PATH = f"{DATASET_PATH}Ground_truth_picture/"
BLUR_IMGS_PATHES = [
    f"{BLUR_PATH}{phone}/Images/" for phone in PHONES]  # include phones

SAVE_PATH='test05/'

def save_images(images, iteration):
    filename = SAVE_PATH + "Iter_" + str(iteration) + ".png"
    torchvision.utils.save_image(images, filename)

def test():
    dataset = SmartDocQADataset(
            blur_image_pathes=BLUR_IMGS_PATHES,

            # We don't need to indicate phone path of sharp image pathes,
            # result in sharp images depend on blur image phone pathes
            sharp_image_root=SHARP_PATH,

            # Other parameters just keeped from GoPro dataset, but no implementation
            center_crop_size=(128,128),
            random_crop_size=128,
            Center_Crop=True,
            Random_Crop = False,
            transform=transforms.Compose([
                transforms.Resize((1164, 1680)),
                transforms.ToTensor()
                
            ]))
    train_dataset, val_set = torch.utils.data.random_split(
            dataset, [int(0.9*len(dataset)), len(dataset) - int(0.9*len(dataset))])
    train_dataloader = DataLoader(
            train_dataset, batch_size=6, pin_memory=True)
    start = 0
    for iteration, images in enumerate(train_dataloader):
        for imgI in range(len(images['blur_image'])):
            save_images(images['blur_image'][imgI]-0.5,f"{iteration}_{imgI}_blur_05")
            save_images(images['sharp_image'][imgI]-0.5, f"{iteration}_{imgI}_gt_05")
            save_images(images['blur_image'][imgI],f"{iteration}_{imgI}_blur_00")
            save_images(images['sharp_image'][imgI], f"{iteration}_{imgI}_gt_00")
test()