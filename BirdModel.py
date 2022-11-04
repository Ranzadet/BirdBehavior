'''
Benen Sullivan 2022
'''

import numpy as np
import pandas as pd
import matplotlib
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2 as cv
import tensorflow as tf
import time

root = "/Users/benen/Comp Sci/Python/FDU_Internship/BirdBehavior"
path = os.path.join(root, "Data/2022-09-27 Chimney Swift Roost - HD 1080p.mov")
cap = cv.VideoCapture(path)


def findBounds(num):
    return (num // 20, num//20 + 1)


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("sheeeeee")
        break


    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    nframe = np.array(gray)
    #nframe.shape => (1080, 1920)
    #SMALL TESTING VALUE FOR LOW RUNTIME
    nframe = np.array(nframe[(len(nframe)//2)-32:(len(nframe)//2)+32, (len(nframe[0])//2)-64:(len(nframe[0])//2)+64])
    #nframe.shape => (64, 128)

    # winSize = (64, 128) #should match the dimensions of nframe
    # blockSize = (16, 16)
    # blockStride = (8, 8)
    # cellSize = (8, 8)
    # nbins = 9
    # derivAperture = 1
    # winSigma = 4.
    # histogramNormType = 0
    # L2HysThreshold = 2.0000000000000001e-01
    # gammaCorrection = 0
    # nlevels = 64
    # hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
    #                         histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    # # compute(img[, winStride[, padding[, locations]]]) -> descriptors
    # winStride = (8, 8)
    # padding = (8, 8)
    # locations = ((30,25), (56,42)) #locations of objects (supervised learning) - currently dummy values
    # hist = hog.compute(nframe, winStride, padding, locations)


    '''HOG IMPLEMENTATION'''
#######################################################################################
    histograms = []
    #These numbers are essentially abitrary- they should be hyperparameterized
    cellWidth = 8
    cellHeight = 8
    numCells = (len(nframe) // cellWidth) * (len(nframe[0]) // cellHeight)
    histograms = []
    for i in range(len(nframe) // cellWidth):
        for j in range(len(nframe[0]) // cellHeight):
            cell = np.array(nframe[i*cellWidth:i*cellWidth+cellWidth, j*cellHeight:j*cellHeight+cellHeight]) #each cell should be an 8X8
            # print(cell.shape)
            gx = cv.Sobel(cell, cv.CV_32F, 1, 0, ksize=3) #3 X 3 sobel filter
            gy = cv.Sobel(cell, cv.CV_32F, 0, 1, ksize=3) #3 X 3 sobel filter
            hist = [0,0,0,0,0,0,0,0,0]
            mag, angle = cv.cartToPolar(gx, gy, angleInDegrees=True)
            mag = np.array(mag)
            angle = np.array(angle)
            for a in range(8):
                for b in range(8):
                    m = mag[a][b]
                    ang = angle[a][b]
                    bounds = findBounds(ang)
                    lower = int(bounds[0])
                    upper = int(bounds[1])
                    hist[upper % 9] += ((ang-lower*20)/20) * m #the bigger the angle, the more it will contribute to upper bound
                    hist[lower % 9] += ((upper*20 - ang)/20) * m #the smaller the angle, the more it will contribute to lower bound
            histograms.append(hist)

    #print(histograms)

    #At this point, histograms should contain every histogram for each cell
    #Next, we need to normalize each block to account for lighting variance
    blockSize = (16, 16)
    blocks = []

    #combine each 9x1 histogram vector in the block into a cells*9 X 1 matrix
    #then, normalize the matrix by dividng each value by the vector's L2 norm (root of sum of squares)
    for i in range((len(nframe)//cellWidth)-1):
        for j in range((len(nframe[0]) // cellHeight)-1):
            block = np.array(histograms[i] + histograms[i+1] + histograms[2] + histograms[j+1])
            L2 = np.sqrt(sum(a**2 for a in block))
            for b in range(len(block)):
                block[b] = block[b] / L2
            blocks.append(block)

    #k = np.sqrt(sum(a**2 for a in vector))
    #for a in range(len(vector)):
    #   vector[a] = vector[a] / k

    #Finally,flatten into a 1-dimensional feature matrix (which is usually fed into a SVM)
    features = np.reshape(blocks, len(blocks) * len(blocks[0]))
    #features.shape = 9 * cellsPerBlock * blocksPerImage
    #print(features.shape)
######################################################################################
    #TODO: IMPLEMENT Neural Network - SUPPORT VECTOR MACHINE (support vector clustering?)
    #Run entire video and write feature vector of each frame directly into a text file / spreadsheet?



######################################################################################


    '''
    FOR GRADIENT VISUALISATION
    '''

    #Calculate gradients
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)

    #convert to vector
    #mag, angle = cv.cartToPolar(gx, gy, angleInDegrees=True)

    gx = cv.convertScaleAbs(gx)
    gy = cv.convertScaleAbs(gy)

    combined = cv.addWeighted(gx, 0.5, gy, 0.5, 0)

    # cv.imshow("Sobel X", gx)
    # cv.imshow("Sobel Y", gy)
    cv.namedWindow("Sobel Combined", cv.WINDOW_NORMAL)
    cv.resizeWindow("Sobel Combined", 1920, 1080)
    cv.imshow("Sobel Combined", combined)


#    cv.imshow('frame', gray)
    if cv.waitKey(10) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
