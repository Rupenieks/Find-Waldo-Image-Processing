import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology
import matplotlib.image as mpimg

wallyImg = cv2.imread("Where.jpg")

def ColourSegmentationHSVRed(img):

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([3, 255, 255])
    mask1 = cv2.inRange(imgHSV, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(imgHSV, lower_red, upper_red)

    # join my masks
    mask = mask1 + mask2

    return mask

def ColourSegmentationHLSWhite(img):

    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    Lchannel = imgHLS[:, :, 1]

    # change 250 to lower numbers to include more values as "white"
    mask = cv2.inRange(Lchannel, 250, 255)

    return mask


def DilateMaskVartically(mask):
    newMask = mask

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 5))
    newMask = cv2.dilate(newMask, kernel, iterations=1)

    return newMask;

def RemoveBigAndSmallObjects(in_mask):

    out_mask = in_mask

    w, h = out_mask
    bufferX = 15
    bufferY = 25

    for x in range(0, w):
        for y in range(0, h):
            pixels = out_mask[x: x + bufferX,
                              y: y + bufferY]



    return out_mask


imgRed = ColourSegmentationHSVRed(wallyImg)
imgWhite = ColourSegmentationHLSWhite(wallyImg)

RedWhiteMask = imgRed + imgWhite
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 4))

combinedROI = DilateMaskVartically(imgRed) + DilateMaskVartically(imgWhite)
# cv2.morphologyEx(combinedROI, cv2.MORPH_OPEN, kernel)


# combinedROI = combinedROI & (imgRed | imgWhite)

# combinedROI = cv2.erode(combinedROI,kernel,iterations = 2)
# combinedROI = cv2.dilate(combinedROI,kernel,iterations = 2)
kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))

combinedROI = cv2.morphologyEx(combinedROI, cv2.MORPH_OPEN, kernel2)

bit_or = cv2.bitwise_or(wallyImg,wallyImg, mask= combinedROI)


# cv2.imshow("opening", combinedROI)
cv2.imshow("combined", bit_or)



key = cv2.waitKey(0)