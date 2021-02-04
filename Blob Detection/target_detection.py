# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 17:59:00 2020

@author: acrog
"""
import numpy as np
import cv2


def resizeImg(img):
    return cv2.resize(img, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)


def findcontours(baseImg):
    contours_img = np.zeros(np.shape(baseImg))

    # im_gauss = cv3.GaussianBlur(baseImg, (5, 5), 0)
    im_blur = cv2.medianBlur(baseImg, 1)
    # thresh = cv2.adaptiveThreshold(im_gauss,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    # thresh = cv2.adaptiveThreshold(im_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    ret, thresh = cv2.threshold(im_blur, 127, 255, 0)
    # get contours

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    dialateValue = 1

    dilate = cv2.dilate(thresh, kernel, iterations=dialateValue)
    # Find contours and draw rectangle
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_area = []
    # calculate area and filter into new array
    for con in contours:
        area = cv2.contourArea(con)
        contours_area.append(con)
        '''if 100 < area < 100000:
            contours_area.append(con)
        '''
    for contour in contours:
        cv2.drawContours(contours_img, contour, -1, (255, 255, 255), 3)

    # detector = cv2.SimpleBlobDetector()

    '''
    keypoints = detector.detect(baseImg)
    
    imgKeypoints = cv2.drawKeypoints(baseImg, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    '''
    return contours_img


def blobDect(baseImg):
    im_blur = cv2.medianBlur(baseImg, 5)
    
    inverted_img = cv2.bitwise_not(im_blur)
    #thresh = cv2.adaptiveThreshold(im_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    # thresh = cv2.adaptiveThreshold(im_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    ret, thresh = cv2.threshold(im_blur, 100, 255, cv2.THRESH_TOZERO)
    #ret, thresh = cv2.threshold(thresh, 127, 255, 0)
    #adaptive_thresh = cv2.adaptiveThreshold(inverted_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #cv2.THRESH_BINARY,11,2)
    # get contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    dialateValue = 1

    dilate = cv2.dilate(thresh, kernel, iterations=dialateValue)
    
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False
    #params.filterByArea = True
    params.maxArea = 100000
    params.minArea = 1000

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(dilate)
    #print("keypoints", keypoints, keypoints[0].pt[0], keypoints[0].pt[1], keypoints[0].size)

    blobs = cv2.drawKeypoints(dilate, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return blobs


parentDir = "C:/Users/acrog/Documents/Purdue/IEEE/MyDataset/Full Images/"
#parentDir = "C:/Users/acrog/Documents/GitHub/PARTIEEE_CV_2020/Image Dataset/Full Image/"
imagePath = parentDir + "circle_#4c507f_N_#63ab78.jpg"

baseImg = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

# crops out the google maps thing at the top
baseImg = baseImg[0:baseImg.shape[1]][200:baseImg.shape[0]]


contourImg = findcontours(baseImg)

blobImg = blobDect(baseImg)

baseImg = resizeImg(baseImg)

contourImg = resizeImg(contourImg)

blobImg = resizeImg(blobImg)

cv2.imshow('contour image', contourImg)
cv2.imshow('base image', baseImg)
cv2.imshow('blob image',blobImg)
cv2.waitKey(0)
cv2.destroyAllWindows()