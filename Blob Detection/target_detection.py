# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 17:59:00 2020

@author: acrog
"""
import numpy as np
import cv2

def resizeImg(img):
    return cv2.resize(img, (0,0), fx=0.4, fy=0.4,interpolation = cv2.INTER_AREA)

def findcontours(baseImg):
    contours_img = np.zeros(np.shape(baseImg))
    
    #im_gauss = cv2.GaussianBlur(baseImg, (5, 5), 0)
    im_blur = cv2.medianBlur(baseImg, 1)
    #thresh = cv2.adaptiveThreshold(im_gauss,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    #thresh = cv2.adaptiveThreshold(im_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    ret, thresh = cv2.threshold(im_blur, 127, 255, 0)
    # get contours
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    
    dialateValue = 1
    
    dilate = cv2.dilate(thresh, kernel, iterations=dialateValue)
    # Find contours and draw rectangle
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_area = []
    # calculate area and filter into new array
    for con in contours:
        area = cv2.contourArea(con)
        if 100 < area < 100000:
            contours_area.append(con)
            
    for contour in contours:
       cv2.drawContours(contours_img, contour, -1, (255, 255, 255), 3)
    
    #detector = cv2.SimpleBlobDetector()
    
    '''
    keypoints = detector.detect(baseImg)
    
    imgKeypoints = cv2.drawKeypoints(baseImg, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    '''
    return contours_img

def blobDect(baseImg):
    detector = cv2.SimpleBlobDetector()
    keypoints = detector.detect(baseImg)
    blank = np.zeros((1,1))
    
    blobs = cv2.drawKeypoints(baseImg, keypoints, blank, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    return blobs

parentDir = "C:/Users/acrog/Documents/Purdue/IEEE/MyDataset/Full Images/"
imagePath = parentDir + "circle_#53a0b3_H_#851ed8.png"

baseImg = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
baseImg = baseImg[0:baseImg.shape[1]][200:baseImg.shape[0]]
print(baseImg)

contourImg = findcontours(baseImg)

#blobImg = blobDect(baseImg)

baseImg = resizeImg(baseImg)

contourImg = resizeImg(contourImg)

#blobImg = resizeImg(blobImg)

cv2.imshow('contour image',contourImg)
cv2.imshow('base image',baseImg)
#cv2.imshow('blob image',blobImg)
cv2.waitKey(0)
cv2.destroyAllWindows()