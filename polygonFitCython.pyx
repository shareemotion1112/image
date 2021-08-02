# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 08:28:05 2021

@author: coolc
"""


# import pyximport 
# pyximport.install() # setup.py 없이 .pyx 파일을 직접 로드할 수 있음..

import numpy as np
cimport numpy as np
import time
import pandas as pd


import tkinter as tk
import tkinter.filedialog as fd
import cv2
import matplotlib.pyplot as plt
import copy
import random

np.import_array()


DTYPE = np.int
ctypedef np.int_t DTYPE_t
ctypedef np.float64_t DTYPE_F_t




# inline 선언은 C컴파일러로 함수를 전달 함(가능한 경우에만)
# cdef inline int int_max(int a, int b): return a if a >= b else b
# cdef inline int int_min(int a, int b): return a if a <= b else b



# cdef extern from "std.h":
#     double standardDeviation(vector[double])

# def standard_dev(lst):
#     # This pre-conversion has some performance improvements.
#     cdef vector[double] v = lst

#     return standardDeviation(v)



# numpy 배열 선언 np.ndarray
def increaseX(np.ndarray point not None, int offset = 1):
    assert point.dtype == DTYPE # point 배열의 값이 int인지 확인
    if point[0] + offset > 0:
        return [point[0] + offset, point[1]]
    else:
        return point
    
def increaseY(np.ndarray point not None, int offset = 1):
    assert point.dtype == DTYPE
    if point[1] + offset > 0:
        return [point[0], point[1] + offset]
    else:
        return point
    
    
 
def getLineProfile(np.ndarray pts, np.ndarray imgArray):        
    
    st = time.time()
    cdef double slope = 0
    cdef double offset = 0
    cdef double y = 0
    cdef int size = 0
        
    # print(pts)
    # print(pts[0])
    newPts = np.append(pts, [pts[0]], axis = 0)
    
    # size check
    for i in range(len(newPts) - 1):                
        p1 = newPts[i]
        p2 = newPts[i + 1]
        
        slope = (p2 - p1)[1]/(p2 - p1)[0]
        offset = p1[1] - slope * p1[0]
        
        if(p1[0] - p2[0]) > 0 :
            startP = p2[0]
            endP = p1[0]
        else:
            startP = p1[0]
            endP = p2[0]
            
        size += 1
        size = size + endP - startP
    
    cdef np.ndarray[DTYPE_t, ndim=1] resultX = np.zeros(size, dtype=np.int32)
    cdef np.ndarray[DTYPE_F_t, ndim=1] resultY = np.zeros(size, dtype=np.float64)
    cdef int index = 0

    # make line
    for i in range(len(newPts) - 1):                
        p1 = newPts[i]
        p2 = newPts[i + 1]
        
        slope = (p2 - p1)[1]/(p2 - p1)[0]
        offset = p1[1] - slope * p1[0]
        
        if(p1[0] - p2[0]) > 0 :
            startP = p2[0]
            endP = p1[0]
        else:
            startP = p1[0]
            endP = p2[0]
        
        for j in range(startP, endP):
            y = slope * j + offset            
            resultX[index] = j
            resultY[index] = y
            index += 1
    end = time.time()
    # print('getLineProfile: ' + str(end -st))
    return resultX, resultY


def getOneMetricForCalStd(np.ndarray imgArray, np.ndarray lineX not None, np.ndarray lineY not None):    
    st = time.time()
    OneMetric = np.zeros_like(imgArray)
    for i in range(imgArray.shape[1]):
        # tmp = lineProfile_df.loc[lineProfile_df['x'] == i]
        tmp = lineY[np.where(lineX == i)]
        if tmp.shape[0] == 2:
            min = np.int32(tmp[0])
            max = np.int32(tmp[1])
            OneMetric[min:max, i] = 1
    end = time.time()
    # print('getOneMetricForCalStd : ' + str(end - st))
    return OneMetric

def getStdOfArea(np.ndarray imgArray not None, np.ndarray OneMetric not None):    
    cdef double nArea
    cdef double st = time.time()
    cdef np.ndarray dotM = imgArray * OneMetric
        
    try :         
        nArea = pd.value_counts(OneMetric.ravel())[1]
    except Exception as ex:
        print('Exception at getStdArea : ' + str(ex))
        nArea = 1
           
    # cdef double std = standard_dev(dotM.ravel())
    cdef double std = np.std(dotM)
    cdef double mean = np.sum(dotM)/nArea 
    cdef double sum = np.sum(OneMetric)
    
    cdef double end = time.time()
    
    # print('getStdOfArea : ' + str(end - st))
    
    return std, mean, sum


def checkLimit(value, limit = 100):
    if (np.abs(value) > limit):
        return value / np.abs(value) * limit
    else : 
        return value
        
    
def getOffset(double old, double new):
    # if np.abs(old - new) == 0:
    #     return 0
    # else:
    #     return (old - new) / np.abs(old - new)
    return (old - new)
    
    
def openFile(basePath):
    root = tk.Tk()
    root.geometry('320x240')
    root.title('open filedialog')
    root.withdraw()
    frame = tk.Frame(root)
    frame.pack()
    
    path = fd.askopenfilename(initialdir = basePath, title = 'select File', filetypes = (("all files", "*.*"), ("image files", "*.tif"), ("text files", "*.txt")))
    
    return path
    
    
# cv2 라이브러리 활용 : cpython 코드 전환시 문제 발생 
def getSegmentedImagByKmeans(imgArray, k = 2):
    pixel_values = np.float32(imgArray.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    __, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()    
    segmentedImage = centers[labels].reshape(imgArray.shape)
    return segmentedImage



    
def cartesianToPolarcoord(DTYPE_t x, DTYPE_t y, np.ndarray centroid not None):
    x = x - centroid[0]
    y = y - centroid[1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return [rho, phi]

def PolarToCartesian(DTYPE_F_t r, DTYPE_F_t p, np.ndarray centroid not None):
    x = np.int32(np.around(r * np.cos(p))) + centroid[0]
    y = np.int32(np.around(r * np.sin(p))) + centroid[1]
    return [x, y]

def getOrderedMatrixByRightClock(np.ndarray originalPoints):    
    centroid = np.array([np.int32( np.mean([o[0] for o in originalPoints]) ), np.int32( np.mean([o[1] for o in originalPoints]) )])
    polarCoord = []
    for o in originalPoints:
        polarCoord.append(cartesianToPolarcoord(o[0], o[1], centroid))
    
    polarSorted = sorted(polarCoord, key=lambda k: k[1])
    
    orderedPoints = []
    for p in polarSorted:
        orderedPoints.append(PolarToCartesian(p[0], p[1], centroid))
    
    return np.array(orderedPoints)



def calRegresssionByAxis(np.ndarray imgArray, np.ndarray pts, int ax, int direction, int ind, np.ndarray offsetLists, np.ndarray lrs, np.ndarray lrsStd, np.ndarray lrsSum, double alphaMean, double alphaStd, double alphaSum ):
    '''
    ax = 0 is x axis
    ax = 1 is y axis
    '''
    
    print( ' point : ' , str(pts[ind]))
    
    if ax == 1:
        print('-----Y----')
        plt.title('Y')
    else : 
        print('---- X ----')
        plt.title('X')
        
        
    st = time.time()
    
    acceleration = 5
    oldP = copy.copy(pts[ind])
    lineX, lineY = getLineProfile(pts, imgArray)
    oldStd, oldMean, oldSum = getStdOfArea(imgArray, getOneMetricForCalStd(imgArray, lineX, lineY))
    
    pts2 = np.concatenate((pts, [pts[0]]), axis = 0)
    # plt.title('old points')
    plt.imshow(imgArray)
    plt.plot(pts2[:, 0],  pts2[:, 1], '--', c='gray')
    plt.scatter(pts[ind][0], pts[ind][1], s = 50, c = 'gray')
    # plt.show()


    offsetValue = offsetLists[ind][ax] * np.random.uniform(-1, 1, 1)[0]
      
    
    if offsetValue == 0:
        offsetValue = 1
    
    
    print('when increase ' + str(offsetValue))

    if ax == 0:
        pts[ind] = increaseX(pts[ind],  offsetValue)
    else : 
        pts[ind] = increaseY(pts[ind],  offsetValue)
    
    newP = copy.copy(pts[ind])
    lineX, lineY = getLineProfile(pts, imgArray)
    newStd, newMean, newSum = getStdOfArea(imgArray, getOneMetricForCalStd(imgArray, lineX, lineY))
    
    
    print('newStd - oldStd  = ' + str(newStd - oldStd) )
    print('newMean - oldMean  = ' + str(newMean - oldMean) )
    
    
    # plt.title('new points')
    # plt.imshow(imgArray)
    
    
    denominator =  oldP[ax]- newP[ax]
    if denominator == 0:
        denominator = 1
        
    offsetStd =    checkLimit( getOffset(newStd, oldStd) * lrsStd[ind][0] * acceleration * alphaStd / denominator  )
    # print('getOffset(oldStd, newStd ) : ' + str(getOffset(oldStd, newStd )))
    print('lrsStd : ' + str(lrsStd[ind][0]))
    # print('denominator : ' + str(denominator))
    # print('alphaStd : ' + str(alphaStd ))
    print('offsetStd : ' + str(offsetStd) )
    
    
    offsetMean = checkLimit( getOffset(oldMean, newMean) * lrs[ind][0] * acceleration  * direction * alphaMean / denominator  )
    # print('getOffset(oldStd, newStd ) : ' + str(  getOffset(oldMean, newMean) ))
    print('lrs : ' + str(lrs[ind][0]))
    # print('denominator : ' + str(denominator))
    # print('alphaMean : ' + str(alphaMean ))        
    print('offsetMean : ' + str(offsetMean))
       
        
    offsetSum = checkLimit( getOffset(oldSum, newSum) * lrsSum[ind][0] * acceleration  * alphaSum / denominator  )
    print('offsetSum : ' + str(offsetSum))
    
    offset = offsetStd + offsetMean + offsetSum
    # offset = offsetStd + offsetMean
    
    if np.abs(offsetMean) > 2 :    
        offsetLists[ind][ax] = offset
        print('result offset : '  + str(offset))
    
    
    if ax == 0:
        pts[ind] = increaseX(oldP, offset)
    else :
        pts[ind] = increaseY(oldP, offset)
    
    pts2 = np.concatenate((pts, [pts[0]]), axis = 0)    
    plt.plot(pts2[:, 0],  pts2[:, 1], c='w')
    plt.scatter(pts[ind][0], pts[ind][1], s = 50, c = 'w')
    plt.show()
    
    # print('new mean : ', newMean, '  old mean : ',  oldMean)
    # print('new std : ', newStd, '  old std : ',  oldStd)
    
    
    if (newMean - oldMean) * direction > 0:
        lrs[ind][ax] = lrs[ind][ax] / Lambda
        # print('lrs[ind][ax] : ' + str(lrs[ind][ax]))
    else :
        lrs[ind][ax] = lrs[ind][ax] * Lambda
        # print('lrs[ind][ax] : ' + str(lrs[ind][ax]))
    
    if newStd > oldStd:
        lrsStd[ind][ax] = lrsStd[ind][ax] * Lambda
        # print('lrsStd[ind][ax] : ' + str(lrsStd[ind][ax]))
    else :
        lrsStd[ind][ax] = lrsStd[ind][ax] / Lambda  
        # print('lrsStd[ind][ax] : ' + str(lrsStd[ind][ax]))
        
    if newSum > oldSum:
        lrsSum[ind][ax] = lrsSum[ind][ax] * Lambda
        # print('lrsSum[ind][ax] : ' + str(lrsSum[ind][ax]))
    else :
        lrsSum[ind][ax] = lrsSum[ind][ax] / Lambda  
        # print('lrsSum[ind][ax] : ' + str(lrsSum[ind][ax]))

    end = time.time()
    # print('calRegresssionByAxis : ' + str(end-st) )
    
    return pts, lrs, lrsStd, lrsSum, newSum, offsetLists



###################### main
###########################


def pologonFit(np.ndarray originalPoints, int imagePath = 0, int iterationLimit = 20, double alphaMean = 1, double alphaStd = 1, double alphaSum = 0.01, int direction = -1):
    '''
    Parameters

    originalPoints : list[list]
        insert initial points atleast 4

    actureArea : the area of fit image
    '''
    st = time.time()
    if imagePath == False:
        imgPath = openFile('C:\\')
    
    
    img = plt.imread(imgPath)
        
    plt.imshow(img);plt.show()
    
    # imgArray = getSegmentedImagByKmeans(img)   
    imgArray = np.array(img[:, :, 0])
    
    pts = getOrderedMatrixByRightClock(originalPoints)
    
    plt.imshow(imgArray); 
    plt.scatter([p[0] for p in pts], [p[1] for p in pts], s = 50, c = 'r')
    plt.show();    
    
    offsetLists = np.ones_like(pts) * 20
    lrs = np.ones_like(pts) * 10
    lrsStd = np.ones_like(pts) * 10
    lrsSum = np.ones_like(pts) * 1e-3    
    global Lambda
    Lambda = 1.2
    
    
    iter = 0
    while iter < iterationLimit:
        iter += 1
        print('-----------------------')
        print(iter)    
        
        for ind, p in enumerate(pts):
            st2 = time.time()
            for k in range(2):
                pts, lrs, lrsStd, lrsSum, newSum, offsetLists = calRegresssionByAxis(imgArray, pts, k, direction, ind, offsetLists, lrs, lrsStd, lrsSum, alphaMean, alphaStd, alphaSum)               
            end2 = time.time()
            print('1 iter time : ' + str(end2 - st2))
            # print(pts)
            
    end = time.time()
    print('total time : ' + str(end - st)  + 's')
    
    plt.title('fitted')
    plt.imshow(imgArray)
    plt.scatter(pts[:, 0], pts[:, 1], s = 50, c = 'w')
    plt.show()
    
    return pts



