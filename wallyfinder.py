import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

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



    return mask;



def FindWaldo(imgRed, imgWhite):

    w, h = imgRed.shape

    bufferX = 15
    bufferY = 30


    for x in range(0, w - bufferX, bufferX):
        for y in range(0, h - bufferY, bufferY):
            pixelsRed = imgRed[x:x + bufferX,
                         y: y + bufferY]

            pixelsWhite = imgWhite[x:x + bufferX,
                         y: y + bufferY]
            redPixelCount = CountPixels(pixelsRed)
            whitePixelCount = CountPixels(pixelsWhite)

            redRatio, whiteRatio = ColourRatio(redPixelCount, whitePixelCount, bufferX, bufferY)





def ColourRatio(count1, count2, bufferX, bufferY):
    totalPixels = bufferX * bufferY
    redRatio = count1 / totalPixels * 100
    whiteRatio = count2 / totalPixels * 100

    # print(redRatio)
    # print(whiteRatio)

    return redRatio, whiteRatio

def CountPixels(pixels):

    h, w = pixels.shape
    pixelCount = 0

    for x in range(0, h):
        for y in range(0, w):
            if(pixels[x, y] != 0):
                pixelCount += 1

    return pixelCount

def DilateMask(mask):

    horizontal = mask

    thresh = cv2.adaptiveThreshold(
        horizontal, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        25,
        15
    )



    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 5))

    # horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # NewMask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, shape)

    return horizontal



imgRed = ColourSegmentationHSVRed(wallyImg)
imgWhite = ColourSegmentationHLSWhite(wallyImg)

dilatedRed = DilateMask(imgRed)
dilatedWhite = DilateMask(imgWhite)

dilatedOverlap = dilatedRed + dilatedWhite

RedWhiteMask = imgRed + imgWhite

bit_or = cv2.bitwise_or(wallyImg,wallyImg, mask= dilatedOverlap);



FindWaldo(imgRed, imgWhite)


plt.imshow(bit_or)
cv2.imshow("combined", bit_or)



key = cv2.waitKey(0)