import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import random
import model2 as models
import torchvision
from torch.utils.data import Dataset, DataLoader
from skimage.measure import compare_psnr
from torchvision import transforms, datasets
from datasets import GoProDataset
import time
from ssim.ssimlib import SSIM
from PIL import Image

parser = argparse.ArgumentParser(description="Deep Multi-Patch Hierarchical Network")
parser.add_argument("-e","--epochs",type = int, default = 2600)
parser.add_argument("-se","--start_epoch",type = int, default = 0)
parser.add_argument("-b","--batchsize",type = int, default = 2)
parser.add_argument("-s","--imagesize",type = int, default = 256)
parser.add_argument("-l","--learning_rate", type = float, default = 0.0001)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()

#Hyper Parameters
METHOD = "DMPHN_1_2_4_Spatial_skip"
SAMPLE_DIR = "test_samples"
EXPDIR = "DMPHN_1_2_4_8_test_res"
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
GPU = args.gpu
BATCH_SIZE = args.batchsize
IMAGE_SIZE = args.imagesize

def save_images(images, name):
    filename = './test_results/' + EXPDIR + "/" + name
    torchvision.utils.save_image(images, filename)

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5*math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())



def main():
    print("init data folders")
    print(METHOD)
    encoder_lv1 = models.Encoder().apply(weight_init).cuda(GPU)
    encoder_lv2 = models.Encoder().apply(weight_init).cuda(GPU)
    encoder_lv3 = models.Encoder().apply(weight_init).cuda(GPU)
    

    decoder_lv1 = models.Decoder().apply(weight_init).cuda(GPU)
    decoder_lv2 = models.Decoder().apply(weight_init).cuda(GPU)
    decoder_lv3 = models.Decoder().apply(weight_init).cuda(GPU)
    

    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")):
        encoder_lv1.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")))
        print("load encoder_lv1 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv2.pkl")):
        encoder_lv2.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv2.pkl")))
        print("load encoder_lv2 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv3.pkl")):
        encoder_lv3.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv3.pkl")))
        print("load encoder_lv3 success")
    
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")):
        decoder_lv1.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")))
        print("load encoder_lv1 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")):
        decoder_lv2.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")))
        print("load decoder_lv2 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv3.pkl")):
        decoder_lv3.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv3.pkl")))
        print("load decoder_lv3 success")
    
    
    if os.path.exists('./test_results/' + EXPDIR) == False:
        os.system('mkdir ./test_results/' + EXPDIR)     
            
    iteration = 0.0
    test_time = 0.0
    total_psnr = 0
    total_ssim = 0
    print("Testing............")
    print("===========================")
    test_dataset = GoProDataset(
        blur_image_files = './datas/GoPro/test_blur_file.txt',
        sharp_image_files = './datas/GoPro/test_sharp_file.txt',
        root_dir = './datas/GoPro',
        transform = transforms.Compose([
        transforms.ToTensor()
    	]))
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False)
    total_psnr = 0
    for iteration, images in enumerate(test_dataloader):
        with torch.no_grad():                                   
            images_lv1 = Variable(images['blur_image'] - 0.5).cuda(GPU)
            start = time.time()
            H = images_lv1.size(2)
            W = images_lv1.size(3)                    
            images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
            images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
            images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
            images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
            images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
            images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]
                    
            feature_lv3_1 = encoder_lv3(images_lv3_1)
            feature_lv3_2 = encoder_lv3(images_lv3_2)
            feature_lv3_3 = encoder_lv3(images_lv3_3)
            feature_lv3_4 = encoder_lv3(images_lv3_4)
            feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
            feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
            feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
            residual_lv3_top = decoder_lv3(feature_lv3_top)
            residual_lv3_bot = decoder_lv3(feature_lv3_bot)
                    
            feature_lv2_1 = encoder_lv2(images_lv2_1 + residual_lv3_top)
            feature_lv2_2 = encoder_lv2(images_lv2_2 + residual_lv3_bot)
            feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
            residual_lv2 = decoder_lv2(feature_lv2)
                    
            feature_lv1 = encoder_lv1(images_lv1 + residual_lv2) + feature_lv2
            deblur_image = decoder_lv1(feature_lv1)

            
            psnr = compare_psnr(images['sharp_image'].numpy()[0], deblur_image.detach().cpu().numpy()[0]+0.5)
            # ssim = SSIM(np.transpose(images['sharp_image'].numpy()[0], (1, 2, 0)), np.transpose(deblur_image.detach().cpu().numpy()[0], (1, 2, 0))+0.5, multichannel=True, data_range=1.0)
            
            pilReal = Image.fromarray(((np.transpose(images['sharp_image'].numpy()[0], (1, 2, 0)))* 255).astype(np.uint8))
            pilFake = Image.fromarray(((np.transpose(deblur_image.detach().cpu().numpy()[0], (1, 2, 0))+0.5)* 255).astype(np.uint8))
            ssim =  SSIM(pilFake).cw_ssim_value(pilReal)

            total_psnr += psnr
            total_ssim += ssim
            stop = time.time()
            test_time += stop-start
            print("testing...... iteration: ", iteration, ", psnr: ", psnr ,", ssim: ", ssim) 
            iteration += 1

    print("PSNR is: ", (total_psnr/1111))
    print("SSIM is: ", (total_ssim/1111))

if __name__ == '__main__':
    main()

        

        

