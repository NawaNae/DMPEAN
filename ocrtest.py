import pytesseract
import cv2
import numpy as np
import argparse
parser = argparse.ArgumentParser(
    description="OCR test")
parser.add_argument("-e", "--epoch", type=int, default=2300)
parser.add_argument("-i", "--iter", type=int, default=2)
parser.add_argument("-d", "--data", type=int, default=1)
args = parser.parse_args()


def ocrCheck(file):

    img = cv2.imread(file)
    img = np.rot90(img, axes=(1, 0))
    # cv2.imwrite("test.png",img)
    fname = file.split("/")[-1]
    print(f"==={fname}===")
    print(pytesseract.image_to_string(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    print()
    return pytesseract.image_to_string(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def checkIter(iter=args.iter, epoch=args.epoch, data=args.data, root="checkpoints/DMPHN_1_2_4_8/", suffix=".png"):
    for test in ['blur', 'deblur', 'gt']:
        ocrCheck(f'{root}epoch{epoch}/Iter_{iter}_{data}_{test}{suffix}')


def ocr_test(filename=''):
    return ocrCheck(filename)


if __name__ == '__main__':
    #checkIter()
    ocrCheck("test_results/DMPHN_1_2_4_8_Center_test_res/testblur.png")

# def test(root=filename):
#     ocrCheck()
