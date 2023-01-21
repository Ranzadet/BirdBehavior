from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
import os
import time

rng.seed(12345)

curr = [1]
curr2 = [1]
curr3 = [1]

def thresh_callback(val):
    threshold = val

    #Reshape cv_gray to (128, 64) ?
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    curr[0] = drawing
    curr2[0] = contours
    curr3[0] = boundRect

    # for i in range(len(contours)):
    #     color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    #     #cv.drawContours(drawing, contours_poly, i, color)
    #     if int(boundRect[i][2]) > 100:
    #         continue
    #     if int(boundRect[i][3]) > 100:
    #         continue
    #     if int(boundRect[i][1] + boundRect[i][3]) > 500:
    #         continue
    #     cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
    #                  (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
    #     #cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
    #
    # cv.namedWindow('Contours')
    # cv.resizeWindow('Contours', 1920, 1080)
    # cv.imshow('Contours', drawing)


parser = argparse.ArgumentParser(description='Code for Creating Bounding boxes and circles for contours tutorial.')
parser.add_argument('--input', help='Path to input image.', default='stuff.jpg')
args = parser.parse_args()
#src = cv.imread(cv.samples.findFile(args.input))
root = "/Users/benen/Comp Sci/Python/FDU_Internship/BirdBehavior"
path = os.path.join(root, "Data/2022-09-27 Chimney Swift Roost - HD 1080p.mov")
cap = cv.VideoCapture(path)

boxArray = []
boxTimer = {}
framecount = 1
starttime = time.time()
lasttime = time.time()
totalBoxes = 0
lastframecount = 1
y = [1,1800*3,1800*6,1800*9,1800*12,1800*15,1800*21,1800*27]
x = y[0]
while cap.isOpened():
#3, 0, 15, 18, 16,
# 12, 6, 1
    #####
    #FOR SKIPPING TO X FRAME
    #####
    if framecount < x:
        ret, frame = cap.read()
        if not ret:
            print("sheeeeee")
            break
        framecount += 1
        continue
    if y.index(x) + 1 < len(y):
        x = y[y.index(x)+1]
    ret, frame = cap.read()
    print("X: ", x, " framecount: ", framecount)


    if not ret:
        print("sheeeeee")
        break

    src_gray = np.array(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))

    max_thresh = 255
    thresh = 100  # initial threshold
    thresh_callback(thresh)

    boxCount = 0
    for i in range(len(curr3[0])):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        #cv.drawContours(drawing, contours_poly, i, color)
        if int(curr3[0][i][2]) > 100 or int(curr3[0][i][2]) < 20:
            continue
        if int(curr3[0][i][3]) > 100 or int(curr3[0][i][3]) < 20:
            continue
        if int(curr3[0][i][1] + curr3[0][i][3]) > 600:
            continue
        ycoord2 = int(curr3[0][i][1] + curr3[0][i][3])
        xcoord2 = int(curr3[0][i][0] + curr3[0][i][2])
        ycoord1 = int(curr3[0][i][0])
        xcoord1 = int(curr3[0][i][0])
        ycenter = max(ycoord1, ycoord2) - min(ycoord1, ycoord2)//2 #center x coordinate of the bounding box
        xcenter = max(xcoord1, xcoord2) - min(xcoord1, xcoord2)//2 #center y coordinate of the bounding box

        #print((xcenter, ycenter))
        colorarray = [0,0,0]
        pixelcount = 0
        foundPixel = False

        for l in range(xcenter-5, xcenter+5):
            for j in range(ycenter-5, ycenter+5):
                colorarray[0] += frame[l][j][0]
                colorarray[1] += frame[l][j][1]
                colorarray[2] += frame[l][j][2]
                currPix = frame[l][j]
                if currPix[0] < 100 or currPix[1] < 100 or currPix[2]:
                    foundPixel = True
                    #print(currPix)
                pixelcount+= 1
        for z in range(len(colorarray)):
            colorarray[z] = colorarray[z] / pixelcount #CHANGE TO FIND SINGLE BIRD PIXEL

        centerColor = colorarray #frame[xcenter][ycenter]
        # if centerColor[0] > 100 and centerColor[1] > 100 and centerColor[2] > 100:
        #     continue
        # if not foundPixel:
        #     continue
        #else, the center color is probably black AND the size of the box is small enough that it is probabyl a bird
        #increment the count of bounding boxes
        #print(colorarray)
        boxCount += 1
        #print(boxCount)

        if boxTimer.get(str(curr3[0][i])) is not None and boxTimer.get(str(curr3[0][i])) > time.time():
            boxTimer[str(curr3[0][i])] += 10
            continue
        boxTimer[str(curr3[0][i])] = time.time() + 10
        boxArray.append(str(curr3[0][i]))

        cv.rectangle(frame, (int(curr3[0][i][0]), int(curr3[0][i][1])), \
                     (int(curr3[0][i][0] + curr3[0][i][2]), int(curr3[0][i][1] + curr3[0][i][3])), color, 2)
        #cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
    source_window = 'Source'
    # if framecount == 1:
    #     cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
    cv.namedWindow(source_window)
    cv.resizeWindow(source_window, 1920, 1080)
    cv.imshow(source_window, frame)
    if cv.waitKey(10) == ord('q'):
        break

    totalBoxes += boxCount
    timeUpdate = 300 #5 minutes
    #print number of birds every timeUpdate seconds
    if (int(time.time()) - int(starttime)) % timeUpdate == 0 and int(lasttime) != int(time.time()):
        print(int(totalBoxes / lastframecount)) #print(boxCount)
        lasttime = time.time()
        lastframecount = 0
        totalBoxes = 0
    framecount += 1
    lastframecount += 1


cap.release()
cv.destroyAllWindows()