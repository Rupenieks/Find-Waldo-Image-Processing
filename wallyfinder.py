import cv2
import numpy as np
import matplotlib.pyplot as plt
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

    # output_img = img.copy()
    # output_img[np.where(mask == 0)] = 0
    #
    # output_hsv = imgHSV.copy()
    # output_hsv[np.where(mask == 0)] = 0


    return mask

def ColourSegmentationHLSWhite(img):

    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    Lchannel = imgHLS[:, :, 1]

    # change 250 to lower numbers to include more values as "white"
    mask = cv2.inRange(Lchannel, 250, 255)



    return mask;


def FindWally(img):

    return img;



imgRed = ColourSegmentationHSVRed(wallyImg)
imgWhite = ColourSegmentationHLSWhite(wallyImg)

RedWhiteMask = imgRed + imgWhite

bit_or = cv2.bitwise_or(wallyImg,wallyImg, mask= RedWhiteMask);
sobely = cv2.Sobel(bit_or,cv2.CV_64F,0,1,ksize=1)

plt.imshow(bit_or)
cv2.imshow("combined", bit_or)
cv2.imshow("horizontal gradient", sobely)


key = cv2.waitKey(0)