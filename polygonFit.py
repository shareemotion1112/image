# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 08:28:05 2021

@author: coolc
"""

import numpy as np
import pandas as pd
import tkinter as tk
import tkinter.filedialog as fd
import cv2
import matplotlib.pyplot as plt
import copy
import random



def increaseX(point, offset = 1):
    if point[0] != 0:
        return [point[0] + offset, point[1]]
    else:
        return point
    
def increaseY(point, offset = 1):
    if point[0] != 0:
        return [point[0], point[1] +  + offset]
    else:
        return point
    
    
def getLineProfile(pts, imgArray, debug = False):
    resultX = []; resultY = []
    newPts = copy.copy(pts)
    newPts.append(pts[0])
    
    for i in range(len(newPts) - 1):        
        if debug :
            print(newPts[i])
            print(newPts[0])
        p1 = np.array(newPts[i], np.int32)        
        p2 = np.array(newPts[i + 1], np.int32)
        
        if (p2 - p1)[0] == 0:
            if p2[1] > p1[1] :
                startP = p1[1]
                endP = p2[1]
            else:
                startP = p2[1]
                endP = p1[1]
                
            for i in range(startP, endP):            
                resultX.append(p1[0])
                resultY.append(i)

        else:                
            slope = (p2 - p1)[1]/(p2 - p1)[0]
            offset = p1[1] - slope * p1[0]
            
            if(p1[0] - p2[0]) > 0 :
                startP = p2[0]
                endP = p1[0]
            else:
                startP = p1[0]
                endP = p2[0]
            for i in range(startP, endP):            
                try:
                    y = slope * i + offset   
                    resultX.append(i)
                    resultY.append( np.int32(y) )
                except Exception as ex :                
                    print(i)
                    print(len(resultY))
                    print(ex)
    
    df = pd.DataFrame({'x' : resultX, 'y' : resultY})
    
    return df, resultX, resultY, newPts


def getOneMetricForCalStd(imgArray, lineProfile_df):
    
    OneMetric = np.zeros_like(imgArray)
    for i in range(imgArray.shape[1]):
        tmp = lineProfile_df.loc[lineProfile_df['x'] == i]
        if tmp.shape[0] == 2:
            min = tmp.iloc[0,1]
            max = tmp.iloc[1,1]
            OneMetric[min:max, i] = 1
            
    return OneMetric


def getStdOfArea(imgArray, OneMetric, debug = False):
    
    dotM = imgArray * OneMetric
    nArea = 0
    
    if debug:
        print(OneMetric)
        print(np.unique(OneMetric, return_counts = True))
        print(np.unique(OneMetric, return_counts = True)[1][1])
        
    try : 
        nArea = np.unique(OneMetric, return_counts = True)[1][1]
    except Exception as ex :
        # print('===== imgArray ====')
        # print(np.unique(imgArray, return_counts = True))
        print('===== OneMetric ====')
        print(np.unique(OneMetric, return_counts = True))
        print(ex)
    
    return np.std(dotM), np.sum(dotM)/nArea, np.sum(OneMetric)

def checkLimit(value, limit = 100):
    if (np.abs(value) > limit):
        return value / np.abs(value) * 10
    elif value == 0:
        return 1
    else : 
        return value
    
def getOffset(old, new):
    if np.abs(old - new) == 0:
        return 0
    else:
        return (old - new) / np.abs(old - new)
    
    
def openFile(basePath):
    root = tk.Tk()
    root.geometry('320x240')
    root.title('open filedialog')
    root.withdraw()
    frame = tk.Frame(root)
    frame.pack()
    
    path = fd.askopenfilename(initialdir = basePath, title = 'select File', filetypes = (("all files", "*.*"), ("image files", "*.tif"), ("text files", "*.txt")))
    
    return path
    
    
def getSegmentedImagByKmeans(imgArray, k = 2):
    pixel_values = imgArray.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    __, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()    
    segmentedImage = centers[labels].reshape(imgArray.shape)
    plt.imshow(segmentedImage)
    return segmentedImage
    
def cartesianToPolarcoord(x, y, centroid):
    x = x - centroid[0]
    y = y - centroid[1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return [rho, phi]

def PolarToCartesian(r, p, centroid):
    x = np.int32(np.around(r * np.cos(p))) + centroid[0]
    y = np.int32(np.around(r * np.sin(p))) + centroid[1]
    return [x, y]

def getOrderedMatrixByRightClock(originalPoints):    
    centroid = [np.int32( np.mean([o[0] for o in originalPoints]) ), np.int32( np.mean([o[1] for o in originalPoints]) )]
    polarCoord = []
    for o in originalPoints:
        polarCoord.append(cartesianToPolarcoord(o[0], o[1], centroid))
    
    polarSorted = sorted(polarCoord, key=lambda k: k[1])
    
    orderedPoints = []
    for p in polarSorted:
        orderedPoints.append(PolarToCartesian(p[0], p[1], centroid))
    
    return orderedPoints



def calRegressionByAxis(imgArray, pts, ax, direction, ind):
    '''
    ax = 0 is x axis
    ax = 1 is y axis
    '''
    
    print( ' point : ' , str(pts[ind]))
    
    oldP = copy.copy(pts[ind])
    lineProfile_df, lineX, lineY, pts = getLineProfile(pts, imgArray)
    oldStd, oldMean, oldSum = getStdOfArea(imgArray, getOneMetricForCalStd(imgArray, lineProfile_df), False)
    
    plt.title('old points')
    plt.imshow(imgArray)
    plt.scatter(lineX, lineY, s = 1)
    plt.scatter(pts[ind][0], pts[ind][1], s = 50, c = 'w')
    plt.show()
    
    if ax == 0:
        pts[ind] = increaseX(pts[ind],  offsetLists[ind][ax])
    else : 
        pts[ind] = increaseY(pts[ind],  offsetLists[ind][ax])
    
    newP = copy.copy(pts[ind])
    lineProfile_df, lineX, lineY, pts = getLineProfile(pts, imgArray)
    newStd, newMean, newSum = getStdOfArea(imgArray, getOneMetricForCalStd(imgArray, lineProfile_df), False)
    
    plt.title('new points')
    plt.imshow(imgArray)
    plt.scatter(lineX, lineY, s = 1)
    plt.scatter(pts[ind][0], pts[ind][1], s = 50, c = 'w')
    plt.show()
    
    denominator = newP[ax] - oldP[ax]
    if denominator == 0:
        denominator = 1
    offset =  checkLimit(  getOffset(newStd, oldStd) * lrsStd[ind][0] * size ) / denominator * alphaStd + \
        checkLimit (getOffset(newMean, oldMean) * lrs[ind][0] * size  ) * direction / denominator * alphaMean
    offsetLists[ind][ax] = offset
    
    print('offset : '  + str(offset))
    print('offset STd : ' , checkLimit( getOffset(newStd, oldStd) * lrsStd[ind][0] * size ) / denominator)
    print('offset Mean : ' , checkLimit( getOffset(newMean, oldMean) * lrs[ind][0] * size  ) * direction / denominator)
    
    if ax == 0:
        pts[ind] = increaseX(oldP, offset)
    else :
        pts[ind] = increaseY(oldP, offset)
        
    print('new mean : ', newMean, '  old mean : ',  oldMean)
    print('new std : ', newStd, '  old std : ',  oldStd)
    
    
    if (newMean - oldMean) * direction > 0:
        lrs[ind][ax] = lrs[ind][ax] / Lambda
    else :
        lrs[ind][ax] = lrs[ind][ax] * Lambda    
    
    if newStd > oldStd:
        lrsStd[ind][ax] = lrsStd[ind][ax] * Lambda
    else :
        lrsStd[ind][ax] = lrsStd[ind][ax] / Lambda    
    
    
    return pts, lrs, newSum



###################### main ##########################
######################################################


def pologonFit(originalPoints, imagePath = False):
    '''
    Parameters

    originalPoints : list[list]
        insert initial points atleast 4

    actureArea : the area of fit image
    '''
   
    
    if imagePath == False:
        imagePath = openFile('C:\\')
    
    
    img = plt.imread(imagePath)
    
    # imgArray = getSegmentedImagByKmeans(img)    
    # if np.unique(imgArray, return_counts = True)[0].shape[0] > 2 :
    #     ret, imgArray = cv2.threshold(img, np.nanmean(img), 255, cv2.THRESH_BINARY)
    #     plt.title('segmented image by threshold')
    #     plt.imshow(imgArray)
    #     plt.show()
    
    imgArray = np.array(img[:,:,0])
    
    pts = getOrderedMatrixByRightClock(originalPoints)
    
    plt.imshow(imgArray)
    plt.scatter([p[0] for p in pts], [p[1] for p in pts], s = 50, c = 'r')
    plt.show()
    
    global offsetLists, lrs, lrsStd,  Lambda, size
    
    
    offsetLists = np.ones_like(pts)
    lrs = np.ones_like(pts)
    lrsStd = np.ones_like(pts)
    Lambda = 2
    length = len(pts)
    
    size = imgArray.shape[0] * imgArray.shape[1]
    
    iter = 0
    while iter < iterationLimit:
        iter += 1
        print('-----------------------')
        print(iter)    
        
        for index in range(length):
            print('index : ' + str(index))
            print('----  X  ----')
            pts, lrs, newSum = calRegressionByAxis(imgArray, pts, 0, direction, index)
        
            
            print('----  Y  ----')
            pts, lrs, newSum = calRegressionByAxis(imgArray, pts, 1, direction, index)
                    





global iterationLimit, direction, alphaMean, alphaStd
iterationLimit = 20
direction = -1 ## 1 is to find bright color, -1 is to find dark color
alphaMean = 5
alphaStd = 2

originalPoints = [[600, 2500], [1600, 2400], [1600, 2600], [600, 2900]]

pologonFit(originalPoints, imagePath='C:/이든 보증서.jpg')


















    