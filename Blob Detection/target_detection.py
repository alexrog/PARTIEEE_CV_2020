# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 17:59:00 2020

@author: acrog
"""
import numpy as np
import cv2

imagePath = "../Image Dataset/Full Image/circle_#0a6d2e_F_#c29faa.png"

x = 1

baseImg = cv2.imread(imagePath)
print(baseImg)
'''
detector = cv2.SimpleBlobDetector()

keypoints = detector.detect(baseImg)

imgKeypoints = cv2.drawKeypoints(baseImg, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

baseImg = cv2.resize(baseImg, (0,0), fx=0.7, fy=0.7,interpolation = cv2.INTER_AREA)
print(len(imgKeypoints))
cv2.imshow('base image',baseImg)
cv2.waitKey(0)
cv2.destroyAllWindows()'''