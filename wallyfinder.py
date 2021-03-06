# RONALDS UPENIEKS
# IMAGE PROCESSING ASSIGNMENT 2
# C15489352 DT282/4

# This program processes in an image from the local directory "Where.jpg" and finds Wally.
# 1. COLOUR SEGMENTATION - BINARY MASKS
#    The process involves the segmentation of colours:
#    (HSV)
#    Bright red (0, 50, 50) to (3, 255, 255) and dark red (170, 50, 50) to (180, 255, 255) are extracted in two separate
#    binary masks from the image and combined afterwards for one binary mask that presents both dark and bright red.
#    (HLS)
#    White is then extracted in a binary mask also, using only one mask as opposed to red, with a light channel.
#
# 2. VERTICAL DILATION OF MASKS
#    Both the red and white masks are vertically dilated with a morphological transformation (using MORPH_CROSS(1,4))
#    to construct the structuring element (kernel). This means that the striped lines of red and white will spread
#    vertically (upward) and overlap each other.
#
#3.  CAPTURING OVERLAPPING PIXELS
#    A new image is created that's completely black. We iterate through it with X and Y while simultaneously checking
#    the same coordinates in both VERTICALLY DILATED masks of white and red. If a red pixel and a white pixel exist
#    in the same coordinates, we create a white pixel on those coordinates on the new black image. This is how we
#    create a new mask that represents where pixels overlap. That's how we locate a potential Wally because of the
#    striped pattern on his jumper.
#
#4.  DILATION AND OPENING - REMOVING SMALL/INVALID ELEMENTS
#    We now dilate the image to strengthen any remnants left, and then run opening to erode smaller elements and
#    further strengthen anything that is left. This leaves us with a small portion of Wally's chest in the mask/
#
#5.  CAPTURING WALLY'S COORDINATES IN ORDER TO DRAW CIRCLE AROUND HIM
#    For the last part of actually marking him out in the original image, we use a less than honourable method of
#    simply locating the first pixels that are not black which we know represent Wally's chest. Returning the coordina-
#    tes we are able to mark out the place he is in and draw the circle around him in the original image.
import cv2
import numpy as np


wallyImg = cv2.imread("Where.jpg")

# In: Image to segment red colour from
# Creates 2 separate masks based on colour range. Applied for both bright and dark red colour ranges.
# Masks are joined and returned as mask.
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

# In: Image to segment white colour from
# Light channel extracted from image to create new mask.
# Returns new mask.
def ColourSegmentationHLSWhite(img):

    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    Lchannel = imgHLS[:, :, 1]

    # change 250 to lower numbers to include more values as "white"
    mask = cv2.inRange(Lchannel, 220, 255)

    return mask


# In: Binary Mask
#  Mask is dilated vertically to thicken horizontal lines of mask
# Returns: new mask
def DilateMaskVertically(mask):
    newMask = mask

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 4))
    newMask = cv2.dilate(newMask, kernel, iterations=1)

    return newMask;

# In: 2 masks
# Black image is created. The two binary masks inserted represent red and white values present in the image respectively.
# Overlapping (done by vertical dilation) is detected by seeing if both red and white are present in the same coordinates
# in which case the pixel coordinates create a white pixel on the new image. This is done to mark out the location of
# Waldo/Wally
# Returns: New image with new pixels created from overlapping pixels in the two inserted masks.
def OverlappingPixels(mask1, mask2):

    h, w = mask1.shape
    blank_image = np.zeros((h, w), np.uint8)

    for x in range(0, h -1):
        for y in range(0, w -1):
            if(mask1[x, y] == 255 and mask2[x, y] == 255):
                blank_image[x, y] = 255

    return blank_image


# In: Image (Supposedly one with everything black except little bits of Wally)
#     Iterates through image to locate non-black pixels.
# Returns: Coordinates of Waldo/Wally
def WallyCoordinates(img):

    h, w = img.shape


    for x in range(0, h -1):
        for y in range(0, w -1):
            if(img[x, y] != 0):
                return y, x

# Show images
def ShowImages():

    cv2.imshow("Red Mask", imgRed)
    cv2.imshow("White Mask", imgWhite)
    cv2.imshow("Combined vertically dilated masks", combinedROI)
    cv2.imshow("Overlapping pixels", overlappingPixelsImg)
    cv2.imshow("Coloured overlapping", colouredOverlappingPixels)
    cv2.imshow("Wally found", wallyImg)

# Step 1: Binary masks created by segmenting colours from image (Red, white)
imgRed = ColourSegmentationHSVRed(wallyImg)
imgWhite = ColourSegmentationHLSWhite(wallyImg)

# Step 2: Dilate both masks vertically so that horizontal lines are exaggerated
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 4))
dilatedVerticalMaskRed = DilateMaskVertically(imgRed)
dilatedVerticalMaskWhite = DilateMaskVertically(imgWhite)

# Step 3: Combine both vertically dilated masks together into one mask
combinedROI = dilatedVerticalMaskRed + dilatedVerticalMaskWhite

# Step 4: Find overlapping pixels from both masks. (Overlapping done by vertical segmentation. Waldos jumper is horizon-
#         tal lines so overlapping should be present. This returns a new image with only overlapping pixels being marked
overlappingPixelsImg = OverlappingPixels(dilatedVerticalMaskRed, dilatedVerticalMaskWhite)

# Step 5: Dilate overlapping image by 3 x 3 kernel
# Kernel for horizontal dilation
kernel2 = np.ones((3,3),np.uint8)
overlappingPixelsImg = cv2.morphologyEx(overlappingPixelsImg, cv2.MORPH_OPEN, kernel2)

# Step 6: Opening on image to remove "specs" and enlarge Waldo/Wally and leave only him
kernelDilate = np.ones((7,5),np.uint8)
overlappingPixelsImg = cv2.morphologyEx(overlappingPixelsImg, cv2.MORPH_OPEN, kernelDilate)

# Display result in colour
colouredOverlappingPixels = cv2.bitwise_or(wallyImg,wallyImg, mask= overlappingPixelsImg)


# Step 7: Draw circle around Wally on original image
wallyX, wallyY = WallyCoordinates(overlappingPixelsImg)
print(wallyX, wallyY)
cv2.circle(wallyImg,(wallyX, wallyY), 30, (255, 0, 0), 3)



# Function to display images
ShowImages()

key = cv2.waitKey(0)