import os
import shutil
import re

match = "[SM]_Img_.*_D\d{1,2}_L\d{1,2}_r35_a-{0,1}\d{1,2}_b-{0,1}\d{1,2}\.jpg"
datasetImgPath="./dataset/smartDocQA/Captured_Images/"
datasetGTImgPath="./dataset/smartDocQA/Ground_truth_picture/"
srcPath,dstPath=[f"{datasetImgPath}Nokia_phone/Images/",f"{datasetImgPath}Samsung_phone/Images/"],[f"{datasetGTImgPath}Nokia_phone/Images/",f"{datasetGTImgPath}Samsung_phone/Images/"]
pathes=zip(srcPath,dstPath)
for srcP,dstP in pathes:
        for fd in os.listdir(srcP):
                if re.search(match,fd):
                        print(f"move {srcP}{fd} to {dstP}{fd}")
                        shutil.move(f"{srcP}{fd}",f"{dstP}{fd}")
        #         print(fd)
        #         shutil.move(f"{path}{fd}",f"../../../Ground_truth_Picture/{fd}")
        #shutil.move("a.py","../../../Ground_truth_Picture/")
