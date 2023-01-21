'''
Benen Sullivan 2022
'''

import numpy as np
import pandas as pd
import matplotlib
import sklearn.model_selection
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2 as cv
import tensorflow as tf
import time
from skimage.io import imread
import skimage
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression


root = "/Users/benen/Comp Sci/Python/FDU_Internship/BirdBehavior"
path = os.path.join(root, "Data/2022-09-27 Chimney Swift Roost - HD 1080p.mov")
cap = cv.VideoCapture(path)

def findBounds(num):
    return (num // 20, num//20 + 1)

frameFile = open("Data/FrameCount.txt", "r")
framenum = frameFile.readline().strip()
if framenum == "":
    framenum = 1
else:
    framenum = int(framenum)
frameFile.close()
localframes = -1

file = open("Data/Features.npy", "wb")

framenum = -1

feature_descriptors = []
test_data = []
select_frames = []
dataset_frames = [0, 30*60*3, 30*60*6, 30*60*9, 30*60*12, 30*60*15, 30*60*21, 30*60*27]
drop_indexes = [0, 6, 12, 18, 24, 30, 42, 54]

while cap.isOpened():
    framenum += 1
    localframes += 1
    if localframes == 79200: #after 44 minutes
        break
    print(localframes)
    if localframes % 900 == 0: #every 30 seconds
        print(localframes)
    elif localframes == 300 or localframes == 600 or localframes == 1200 or localframes == 1500:
        ret, frame = cap.read()
        if not ret:
            print("sheeeeee")
            break
        gframe = np.array(frame)
        nframe = np.array(skimage.transform.resize(gframe, (128 * 8, 64 * 8)))
        fd, hog_image = hog(nframe, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, channel_axis=2)
        test_data.append(fd)
        print(localframes)
        continue
    else:
        ret, frame = cap.read()
        if not ret:
            print("sheeeeee")
            break
        continue


    # if localframes < framenum:
    #     print(localframes)
    #     localframes += 1
    #     continue
    ret, frame = cap.read()

    if not ret:
        print("sheeeeee")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    gframe = np.array(frame)
    # nframe.shape => (1080, 1920)
    #SMALL TESTING VALUE FOR LOW RUNTIME
    #nframe = np.array(nframe[(len(nframe)//2)-32:(len(nframe)//2)+32, (len(nframe[0])//2)-64:(len(nframe[0])//2)+64])
    nframe = np.array(skimage.transform.resize(gframe, (128*8, 64*8)))
    # nframe.shape => (64, 128)



    # winSize = (128*4, 64*4) #should match the dimensions of nframe
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



#
#     '''HOG IMPLEMENTATION'''
# #######################################################################################
#     histograms = []
#     #These numbers are essentially abitrary- they should be hyperparameterized
#     cellWidth = 8
#     cellHeight = 8
#     numCells = (len(nframe) // cellWidth) * (len(nframe[0]) // cellHeight)
#     histograms = []
#     for i in range(len(nframe) // cellWidth):
#         for j in range(len(nframe[0]) // cellHeight):
#             cell = np.array(nframe[i*cellWidth:i*cellWidth+cellWidth, j*cellHeight:j*cellHeight+cellHeight]) #each cell should be an 8X8
#             # print(cell.shape)
#             gx = cv.Sobel(cell, cv.CV_32F, 1, 0, ksize=3) #3 X 3 sobel filter
#             gy = cv.Sobel(cell, cv.CV_32F, 0, 1, ksize=3) #3 X 3 sobel filter
#             hist = [0,0,0,0,0,0,0,0,0]
#             mag, angle = cv.cartToPolar(gx, gy, angleInDegrees=True)
#             mag = np.array(mag)
#             angle = np.array(angle)
#             for a in range(8):
#                 for b in range(8):
#                     m = mag[a][b]
#                     ang = angle[a][b]
#                     bounds = findBounds(ang)
#                     lower = int(bounds[0])
#                     upper = int(bounds[1])
#                     hist[upper % 9] += ((ang-lower*20)/20) * m #the bigger the angle, the more it will contribute to upper bound
#                     hist[lower % 9] += ((upper*20 - ang)/20) * m #the smaller the angle, the more it will contribute to lower bound
#             histograms.append(hist)
#
#     #At this point, histograms should contain every histogram for each cell
#     #Next, we need to normalize each block to account for lighting variance
#     blockSize = (16, 16)
#     blocks = []
#
#     #combine each 9x1 histogram vector in the block into a cells*9 X 1 matrix
#     #then, normalize the matrix by dividng each value by the vector's L2 norm (root of sum of squares)
#     for i in range((len(nframe)//cellWidth)-1):
#         for j in range((len(nframe[0]) // cellHeight)-1):
#             block = np.array(histograms[i] + histograms[i+1] + histograms[2] + histograms[j+1])
#             L2 = np.sqrt(sum(a**2 for a in block))
#             for b in range(len(block)):
#                 block[b] = block[b] / L2
#             blocks.append(block)
#
#     #Finally,flatten into a 1-dimensional feature matrix (which is usually fed into a SVM)
#     features = np.reshape(blocks, len(blocks) * len(blocks[0]))
#
#     # features.shape will have dimensions: 9 * cellsPerBlock * blocksPerImage
#
#     '''
#
#     NEXT: Write the features into a numpy binary
#
#     '''
#
#     opentype = "wb"
#
#     arr = []
#     if (framenum == 1):
#         arr = np.array([])
#     else:
#         arr = np.load("Data/Features.npy")
#     np.append(arr, features)
#     np.save(file, arr)
#
#     print(framenum)
#     frameFile = open("Data/FrameCount.txt", "w")
#     frameFile.write(str(framenum))
#     frameFile.close()
#
#     if framenum == 6:
#         file.close()
#         break
######################################################################################
    #TODO: IMPLEMENT Neural Network - SUPPORT VECTOR MACHINE (support vector clustering?)
    #Run entire video and write feature vector of each frame directly into a text file / spreadsheet?



######################################################################################

    '''
    Saving Hog Descriptors
    '''

    print(nframe.shape)
    #print(nframe)
    fd, hog_image = hog(nframe, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, channel_axis=2) #channel axis change?

    if localframes in dataset_frames:
        select_frames.append(fd)
        continue
    feature_descriptors.append(fd)

    '''
    FOR GRADIENT VISUALISATION
    '''
    #
    # #Calculate gradients
    # gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    # gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    #
    # #convert to vector
    # #mag, angle = cv.cartToPolar(gx, gy, angleInDegrees=True)
    #
    # gx = cv.convertScaleAbs(gx)
    # gy = cv.convertScaleAbs(gy)
    #
    # combined = cv.addWeighted(gx, 0.5, gy, 0.5, 0)
    #
    # cv.imshow("Sobel X", gx)
    # cv.imshow("Sobel Y", gy)


    # cv.namedWindow("Counting", cv.WINDOW_NORMAL)
    # cv.resizeWindow("Counting", 1920, 1080)
    # cv.imshow("Counting", frame)
    # #time.sleep(3)
    #
    # if cv.waitKey(10) == ord('q'):
    #     break

# data1, data2, data3, data4, data5 = "", '', '', '', ''
# with open("Data/Features.npy", "rb") as f:
#     data1 = np.load(f)
#     data2 = np.load(f)
#     data3 = np.load(f)
#     data4 = np.load(f)
#     data5 = np.load(f)
# print(data1)
# print(data2)
# print(data3)
# print(data4)
# print(data5)


X = np.array(feature_descriptors)
counts = []

countFile = open("Data/BirdCounts.txt", "r")
iter_var = 0
for line in countFile:
    line = line.strip()
    if iter_var not in drop_indexes:
        counts.append(int(line))
    iter_var += 1

y = np.array(counts)

print(X.shape)
print(y.shape)
print(y)

regr = make_pipeline(StandardScaler(),
                     LinearSVR(random_state=0, tol=1e-5))
regr.fit(X, y)
print("SVM predictions [model]: ", regr.predict(test_data))
print("SVM predictions [select frames]: ", regr.predict(select_frames)) #predictions at 3, 6, 9, 12, 15, 21, 27 minutes

print(X[0])

#test_data contains frames at 10, 20, 40, and 50 seconds. Predictions should match [6, 8, 4, 10]



cap.release()
cv.destroyAllWindows()
