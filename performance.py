


from PIL import Image
import os
import glob
import numpy as np
from os import listdir
from os.path import isfile, isdir, join


mypath = "/home/gazella/Desktop/johnny/DMPHN-cvpr19-master-master/checkpoints/DMPHN_1_2_4_8/epoch0"


files = listdir(mypath)

for f in files:
  fullpath = join(mypath, f)
  # 判斷 fullpath 是檔案還是目錄
  if isfile(fullpath):
    print("檔案：", f)
  elif isdir(fullpath):
    print("目錄：", f)



def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
