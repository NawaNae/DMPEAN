import numpy as np
import os
import io
import re
import cv2
from pathlib import Path
from collections import Counter 
#DEBUG
DEBUG=False
PRINT_NUM_PER_PHONE=-1#all
PRINT_NUM_PER_PAGE=-1
LIGHT_TEST=True
LIGHT_TEST_LIST=[2,3,4]
PAGE_TEST=False
PAGE_TEST_LIST=[21,22,23,24,25,26,27,28,29,30]
NO_MATCH_P="./cropError/"
WIERD_HW_P="./wierdHW/"

PHONES=["Samsung_phone","Nokia_phone"]
IN_P="./datasetOrigin/smartDocQA/"
OUT_P="./dataset/smartDocQA/"
def show(img,delay=0,title="img"):
    img_s=cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4)))
    cv2.imshow('',img_s)
    cv2.waitKey(delay)
def RGB(r,g,b):
    return np.array([b,g,r])
for rootP in ["Captured_Images/","Ground_truth_picture/"]:
    for phone in PHONES:
        subPath=f"{rootP}{phone}/Images/"
        path=f"{IN_P}{subPath}"
        printLimit=-1
        pageLimit=1
        pageCount=Counter()
        for fname in os.listdir(path):#filter(lambda p:re.match(".*\.jpg",p),os.listdir(path))
            page=0
            light=0
            if re.search(r"D(\d{2})",fname):
                page=int(re.search(r"D(\d{2})",fname).group(1))
            if re.search(r"L(\d{1})",fname):
                light=int(re.search(r"L(\d{1})",fname).group(1))
            if DEBUG:
                if LIGHT_TEST and not light in LIGHT_TEST_LIST:continue
                if PAGE_TEST and not page in PAGE_TEST_LIST:continue
                if PRINT_NUM_PER_PAGE!=-1 and pageCount[page]>=pageLimit:continue
                pageCount[page]+=1
                printLimit-=1
                if printLimit==0:break
            fullPath=f"{path}{fname}"
            imgO=cv2.imread(fullPath)

            h,w,c=imgO.shape
            area=h*w
            
            count=0
            img=imgO.copy()
            noMatch=None
            while count==0: 
            
                maxColor=np.max(img,axis=2)
                minColor=np.min(img,axis=2)
                maxColori=np.argmax(img,axis=2)
                colorDiff=maxColor-minColor
                
                _,colorDiff_mask=cv2.threshold(colorDiff,[40,60][(page>20 and light in [2,3]) or fname in ["S_Img_Android_D2_L1_r35_a0_b0.jpg","S_Img_WP_D5_L3_r35_a0_b0.jpg"] ],255,1)
                #show(colorDiff,title=f"colorDiff_{fname}")
                
                img=cv2.bitwise_and(img, img, mask = colorDiff_mask)
                if noMatch and DEBUG:
                    show(img)
                imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                _, thresh = cv2.threshold(src=imgray, thresh=50, maxval=100, type=0)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
                

                count=0 
                areaLB=area*[0.3,0.03][page>20]
                for sx,sy,sw,sh,sa in stats:
                    sa=sw*sh
                    
                    if sa>areaLB and sa<area*0.9 and sw/sh<3 and sw/sh>=1  :
                        #print(f"w/h : {sw/sh}")

                        if DEBUG:
                            pass
                            #img=cv2.rectangle(img,(sx,sy),(sx+sw,sy+sh),(150,100,100),7)
                            #show(img,delay=1)
                        

                        ansImg=np.ones([h,w,c],np.uint8)
                        rex,rey=int(w/2-sw/2),int(h/2-sh/2)
                        imgOCrop=imgO[sy:(sy+sh),sx:(sx+sw)]
                        ansImg[rey:(rey+sh),rex:(rex+sw)]=imgOCrop
                        if DEBUG:
                            show(ansImg,delay=1)
                        else:
                            Path(f"{OUT_P}{subPath}").mkdir(exist_ok=True,parents=True)
                            cv2.imwrite(f"{OUT_P}{subPath}{fname}",ansImg)

                        if sw/sh>3 or sw/sh<1:
                            Path(f"{WIERD_HW_P}{subPath}").mkdir(exist_ok=True,parents=True)
                            cv2.imwrite(f"{WIERD_HW_P}{subPath}{fname}",imgO)
                            cv2.imwrite(f"{WIERD_HW_P}{subPath}{fname[:-4]}_img.jpg",img)
                        count+=1
                if count==0 and noMatch:
                    print(f"[Can't find document in image by cv in {fname}]")
                    noMatch=None
                    Path(f"{NO_MATCH_P}{subPath}").mkdir(exist_ok=True,parents=True)
                    cv2.imwrite(f"{NO_MATCH_P}{subPath}{fname}",imgO)
                    
                    break
                if count==0:
                    noMatch=True
                    if DEBUG:pass
                    img=imgO.copy()
                    img[:,:,2]=255*(1-np.exp((img[:,:,2])/-49))
                    img[:,:,1]=255*(1-np.exp((img[:,:,1])/-49))
                        

        

        
        
        
