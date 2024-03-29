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
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from datasets import SmartDocQADataset
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import shutil
fileTest = './runs'
if os.path.exists(fileTest):
    shutil.rmtree(fileTest)
writer = SummaryWriter()
parser = argparse.ArgumentParser(
    description="Deep Multi-Patch Hierarchical Network")
parser.add_argument("-e", "--epochs", type=int, default=2400)
parser.add_argument("-se", "--start_epoch", type=int, default=0)
parser.add_argument("-b", "--batchsize", type=int, default=6)
parser.add_argument("-s", "--imagesize", type=int, default=256)
parser.add_argument("-l", "--learning_rate", type=float, default=0.0001)
parser.add_argument("-g", "--gpu", type=int, default=0)
args = parser.parse_args()

# Hyper Parameters
METHOD = "DMPHN_1_2_4_Spatial_skip"
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
GPU = args.gpu
BATCH_SIZE = args.batchsize
IMAGE_SIZE = args.imagesize

# Dataset
DATASET_PATH = "./dataset/smartDocQA/"
PHONES = ["Nokia_phone", "Samsung_phone"]
BLUR_PATH = f"{DATASET_PATH}Captured_Images/"
SHARP_PATH = f"{DATASET_PATH}Ground_truth_picture/"
BLUR_IMGS_PATHES = [
    f"{BLUR_PATH}{phone}/Images/" for phone in PHONES]  # include phones


def save_deblur_images(images, iteration, epoch):
    filename = './checkpoints/' + METHOD + "/epoch" + \
        str(epoch) + "/" + "Iter_" + str(iteration) + "_deblur.png"
    torchvision.utils.save_image(images, filename)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5*math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    if classname.find('Conv1d') != -1:
        n = m.kernel_size[0] * m.out_channels
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


def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def testDataset():
    train_dataset = SmartDocQADataset(
        blur_image_pathes=BLUR_IMGS_PATHES,

        # We don't need to indicate phone path of sharp image pathes,
        # result in sharp images depend on blur image phone pathes
        sharp_image_root=SHARP_PATH,

        # Other parameters just keeped from GoPro dataset, but no implementation

        crop=True,
        crop_size=IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((1280, 720)),
            transforms.CenterCrop(256),
        ]))
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8)
    start = 0
    for iteration, images in enumerate(train_dataloader):
        if iteration % 50 == 0:
            print(iteration)


def main():
    # testDataset()
    # input("dataset test end")
    print("init data folders")

    psnr_list = []
    encoder_lv1 = models.Encoder()
    encoder_lv2 = models.Encoder()
    encoder_lv3 = models.Encoder()

    decoder_lv1 = models.Decoder()
    decoder_lv2 = models.Decoder()
    decoder_lv3 = models.Decoder()

    encoder_lv1.apply(weight_init).cuda(GPU)
    encoder_lv2.apply(weight_init).cuda(GPU)
    encoder_lv3.apply(weight_init).cuda(GPU)

    decoder_lv1.apply(weight_init).cuda(GPU)
    decoder_lv2.apply(weight_init).cuda(GPU)
    decoder_lv3.apply(weight_init).cuda(GPU)

    encoder_lv1_optim = torch.optim.Adam(
        encoder_lv1.parameters(), lr=LEARNING_RATE)
    encoder_lv1_scheduler = StepLR(
        encoder_lv1_optim, step_size=1000, gamma=0.1)
    encoder_lv2_optim = torch.optim.Adam(
        encoder_lv2.parameters(), lr=LEARNING_RATE)
    encoder_lv2_scheduler = StepLR(
        encoder_lv2_optim, step_size=1000, gamma=0.1)
    encoder_lv3_optim = torch.optim.Adam(
        encoder_lv3.parameters(), lr=LEARNING_RATE)
    encoder_lv3_scheduler = StepLR(
        encoder_lv3_optim, step_size=1000, gamma=0.1)

    decoder_lv1_optim = torch.optim.Adam(
        decoder_lv1.parameters(), lr=LEARNING_RATE)
    decoder_lv1_scheduler = StepLR(
        decoder_lv1_optim, step_size=1000, gamma=0.1)
    decoder_lv2_optim = torch.optim.Adam(
        decoder_lv2.parameters(), lr=LEARNING_RATE)
    decoder_lv2_scheduler = StepLR(
        decoder_lv2_optim, step_size=1000, gamma=0.1)
    decoder_lv3_optim = torch.optim.Adam(
        decoder_lv3.parameters(), lr=LEARNING_RATE)
    decoder_lv3_scheduler = StepLR(
        decoder_lv3_optim, step_size=1000, gamma=0.1)

    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")):
        encoder_lv1.load_state_dict(torch.load(
            str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")))
        print("load encoder_lv1 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv2.pkl")):
        encoder_lv2.load_state_dict(torch.load(
            str('./checkpoints/' + METHOD + "/encoder_lv2.pkl")))
        print("load encoder_lv2 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv3.pkl")):
        encoder_lv3.load_state_dict(torch.load(
            str('./checkpoints/' + METHOD + "/encoder_lv3.pkl")))
        print("load encoder_lv3 success")

    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")):
        decoder_lv1.load_state_dict(torch.load(
            str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")))
        print("load encoder_lv1 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")):
        decoder_lv2.load_state_dict(torch.load(
            str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")))
        print("load decoder_lv2 success")
    if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv3.pkl")):
        decoder_lv3.load_state_dict(torch.load(
            str('./checkpoints/' + METHOD + "/decoder_lv3.pkl")))
        print("load decoder_lv3 success")

    if os.path.exists('./checkpoints/' + METHOD) == False:
        os.mkdir('./checkpoints/' + METHOD)

    for epoch in range(args.start_epoch, EPOCHS):
        encoder_lv1_scheduler.step(epoch)
        encoder_lv2_scheduler.step(epoch)
        encoder_lv3_scheduler.step(epoch)

        decoder_lv1_scheduler.step(epoch)
        decoder_lv2_scheduler.step(epoch)
        decoder_lv3_scheduler.step(epoch)

        print("Training...")

        dataset = SmartDocQADataset(
            blur_image_pathes=BLUR_IMGS_PATHES,

            # We don't need to indicate phone path of sharp image pathes,
            # result in sharp images depend on blur image phone pathes
            sharp_image_root=SHARP_PATH,

            # Other parameters just keeped from GoPro dataset, but no implementation

            crop=True,
            crop_size=IMAGE_SIZE,
            transform=transforms.Compose([
                transforms.Resize((1280, 720)),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
            ]))
        uploader = transforms.ToPILImage()
        print(len(dataset))
        train_dataset, val_set = torch.utils.data.random_split(
            dataset, [int(0.9*len(dataset)), len(dataset) - int(0.9*len(dataset))])
        train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
        start = 0
        # for iteration, images in enumerate(train_dataloader):
        #     image_ = uploader(images['blur_image'][0])
        #     plt.imshow(image_)
        #     plt.show()
        #     image_ = uploader(images['sharp_image'][0])
        #     plt.imshow(image_)
        #     plt.show()
        # train_dataset = GoProDataset(
        #     blur_image_files = './datas/GoPro/train_blur_file.txt',
        #     sharp_image_files = './datas/GoPro/train_sharp_file.txt',
        #     root_dir = './datas/GoPro/',
        #     crop = True,
        #     crop_size = IMAGE_SIZE,
        #     transform = transforms.Compose([
        #         transforms.ToTensor()
        #         ]))
        # train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
        # start = 0

        for iteration, images in enumerate(train_dataloader):
            mse = nn.MSELoss().cuda(GPU)

            gt = Variable(images['sharp_image'] - 0.5).cuda(GPU)
            H = gt.size(2)
            W = gt.size(3)

            images_lv1 = Variable(images['blur_image'] - 0.5).cuda(GPU)
            images_lv2_1 = images_lv1[:, :, 0:int(H/2), :]
            images_lv2_2 = images_lv1[:, :, int(H/2):H, :]
            images_lv3_1 = images_lv2_1[:, :, :, 0:int(W/2)]
            images_lv3_2 = images_lv2_1[:, :, :, int(W/2):W]
            images_lv3_3 = images_lv2_2[:, :, :, 0:int(W/2)]
            images_lv3_4 = images_lv2_2[:, :, :, int(W/2):W]

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
            feature_lv2 = torch.cat(
                (feature_lv2_1, feature_lv2_2), 2) + feature_lv3
            residual_lv2 = decoder_lv2(feature_lv2)

            feature_lv1 = encoder_lv1(images_lv1 + residual_lv2) + feature_lv2
            deblur_image = decoder_lv1(feature_lv1)

            loss_lv1 = mse(deblur_image, gt)

            loss = loss_lv1

            encoder_lv1.zero_grad()
            encoder_lv2.zero_grad()
            encoder_lv3.zero_grad()

            decoder_lv1.zero_grad()
            decoder_lv2.zero_grad()
            decoder_lv3.zero_grad()

            loss.backward()

            encoder_lv1_optim.step()
            encoder_lv2_optim.step()
            encoder_lv3_optim.step()

            decoder_lv1_optim.step()
            decoder_lv2_optim.step()
            decoder_lv3_optim.step()

            if (iteration+1) % 50 == 0:
                stop = time.time()
                print("epoch:", epoch, "iteration:", iteration+1,
                      "loss:%.4f" % loss.item(), 'time:%.4f' % (stop-start))
                start = time.time()
                writer.add_scalar("Loss/train", loss.item(),
                                  iteration / 50 + len(train_dataloader) * epoch)
                writer.flush()

        if (epoch) % 100 == 0:
            if os.path.exists('./checkpoints/' + METHOD + '/epoch' + str(epoch)) == False:
                os.system('mkdir ./checkpoints/' +
                          METHOD + '/epoch' + str(epoch))

            print("Testing...")
            # test_dataset = GoProDataset(
            #     blur_image_files = './datas/GoPro/test_blur_file.txt',
            #     sharp_image_files = './datas/GoPro/test_sharp_file.txt',
            #     root_dir = './datas/GoPro/',
            #     transform = transforms.Compose([
            #         transforms.ToTensor()
            #     ]))
            test_dataloader = DataLoader(
                val_set, batch_size=1, shuffle=True, pin_memory=True, num_workers=8)

            #test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False)
            test_time = 0.0
            total_psnr = 0
            for iteration, images in enumerate(test_dataloader):
                with torch.no_grad():
                    images_lv1 = Variable(images['blur_image'] - 0.5).cuda(GPU)
                    start = time.time()
                    H = images_lv1.size(2)
                    W = images_lv1.size(3)
                    images_lv2_1 = images_lv1[:, :, 0:int(H/2), :]
                    images_lv2_2 = images_lv1[:, :, int(H/2):H, :]
                    images_lv3_1 = images_lv2_1[:, :, :, 0:int(W/2)]
                    images_lv3_2 = images_lv2_1[:, :, :, int(W/2):W]
                    images_lv3_3 = images_lv2_2[:, :, :, 0:int(W/2)]
                    images_lv3_4 = images_lv2_2[:, :, :, int(W/2):W]

                    feature_lv3_1 = encoder_lv3(images_lv3_1)
                    feature_lv3_2 = encoder_lv3(images_lv3_2)
                    feature_lv3_3 = encoder_lv3(images_lv3_3)
                    feature_lv3_4 = encoder_lv3(images_lv3_4)
                    feature_lv3_top = torch.cat(
                        (feature_lv3_1, feature_lv3_2), 3)
                    feature_lv3_bot = torch.cat(
                        (feature_lv3_3, feature_lv3_4), 3)
                    feature_lv3 = torch.cat(
                        (feature_lv3_top, feature_lv3_bot), 2)
                    residual_lv3_top = decoder_lv3(feature_lv3_top)
                    residual_lv3_bot = decoder_lv3(feature_lv3_bot)

                    feature_lv2_1 = encoder_lv2(
                        images_lv2_1 + residual_lv3_top)
                    feature_lv2_2 = encoder_lv2(
                        images_lv2_2 + residual_lv3_bot)
                    feature_lv2 = torch.cat(
                        (feature_lv2_1, feature_lv2_2), 2) + feature_lv3
                    residual_lv2 = decoder_lv2(feature_lv2)

                    feature_lv1 = encoder_lv1(
                        images_lv1 + residual_lv2) + feature_lv2
                    deblur_image = decoder_lv1(feature_lv1)

                    stop = time.time()
                    test_time += stop - start
                    psnr = compare_psnr(images['sharp_image'].numpy()[
                                        0], deblur_image.detach().cpu().numpy()[0]+0.5)
                    total_psnr += psnr
                    if (iteration+1) % 50 == 0:
                        print('PSNR:%.4f' % (psnr), '  Average PSNR:%.4f' %
                              (total_psnr/(iteration+1)))
                    save_deblur_images(deblur_image.data +
                                       0.5, iteration, epoch)

            psnr_list.append(total_psnr/(iteration+1))
            print("PSNR list:")
            print(psnr_list)

        torch.save(encoder_lv1.state_dict(), str(
            './checkpoints/' + METHOD + "/encoder_lv1.pkl"))
        torch.save(encoder_lv2.state_dict(), str(
            './checkpoints/' + METHOD + "/encoder_lv2.pkl"))
        torch.save(encoder_lv3.state_dict(), str(
            './checkpoints/' + METHOD + "/encoder_lv3.pkl"))

        torch.save(decoder_lv1.state_dict(), str(
            './checkpoints/' + METHOD + "/decoder_lv1.pkl"))
        torch.save(decoder_lv2.state_dict(), str(
            './checkpoints/' + METHOD + "/decoder_lv2.pkl"))
        torch.save(decoder_lv3.state_dict(), str(
            './checkpoints/' + METHOD + "/decoder_lv3.pkl"))


if __name__ == '__main__':
    main()
