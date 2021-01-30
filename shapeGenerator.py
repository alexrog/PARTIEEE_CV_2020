# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:39:41 2020

@author: acrog
"""

import cv2 
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
import random
import string
import webcolors


def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    
    #noise = np.zeros((image.shape[0], image.shape[1], image.shape[2]))

    #cv2.randu(noise, 0, 150)

    '''b_channel, g_channel, r_channel = cv2.split(noise)

    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 #creating a dummy alpha channel image.

    noise = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))'''
    #noisy_image = image + np.array(0.4*noise, dtype=np.int)
    noisy_image = cv2.blur(image,(1,1))

    return noisy_image

def addTarget(target, background, x, y):
    offset = 5
    temp_background = base_img[x-offset:x+target.shape[0]+offset, y-offset:y+target.shape[1]+offset].copy()
    b_channel, g_channel, r_channel = cv2.split(temp_background)

    alpha_channel = np.zeros(b_channel.shape, dtype=b_channel.dtype) #creating a dummy alpha channel image.

    temp_background = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    
    for i in range(offset,temp_background.shape[0]-offset):
        for j in range(offset,temp_background.shape[1]-offset):
            if target[i-offset][j-offset][3] != 0:
                temp_background[i][j] = target[i-offset][j-offset]
                
    temp_background = cv2.blur(temp_background,(3,3)) 
              
      
    for i in range(temp_background.shape[0]):
        for j in range(temp_background.shape[1]):
            if temp_background[i][j][3] != 0:
                background[x+i][y+j] = temp_background[i][j][0:3]
    
    cv2.imwrite('temp_background.png', background[x-75:x+75, y-100:y+100])   
    return temp_background.shape[0], temp_background.shape[1]
                
def createCircleTarget():
    color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
    hexColors = ["circle",webcolors.rgb_to_hex(color)]
    W, H = random.randint(400,600), random.randint(400,600)
    shape_img = Image.new('RGBA', (W, H), (color[0], color[1], color[2], 0))
    
    draw = ImageDraw.Draw(shape_img)
    draw.ellipse((0, 0, W, H), fill=(color[0], color[1], color[2]))
    font = ImageFont.truetype("Helvetica.ttf", random.randint(250,450))
    # draw.text((x, y),"Sample Text",(r,g,b))
    color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
    
    msg = random.choice(string.ascii_letters)
    hexColors.append(msg)
    hexColors.append(webcolors.rgb_to_hex(color))
    
    w, h = draw.textsize(msg, font=font)
    draw.text(((W-w)/2,(H-h)/2), msg,(color[0],color[1],color[2]),font=font)
    #shape_img= shape_img.rotate(random.randint(0,360))
    shape_img.save('test.png', 'PNG')
    return hexColors
    
def createTriangleTarget():
    color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
    hexColors = ["triangle",webcolors.rgb_to_hex(color)]
    
    t1, t2, t4 = random.randint(0,100), random.randint(0,100), random.randint(0,100)
    t3, t6 = random.randint(400,600), random.randint(400,600)
    t5 = (t3-t1)/2 + random.randint(-10,10)
    W, H = 600, 600
    shape_img = Image.new('RGBA', (W, H), (color[0], color[1], color[2], 0))
    
    draw = ImageDraw.Draw(shape_img)
    draw.polygon(((t1,t2),(t3,t4),(t5,t6)), fill=(color))
    font = ImageFont.truetype("Helvetica.ttf", int((t3-t1+t6-t2)/2)-200)
    # draw.text((x, y),"Sample Text",(r,g,b))
    color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
    
    msg = random.choice(string.ascii_letters)
    hexColors.append(msg)
    hexColors.append(webcolors.rgb_to_hex(color))
    
    w, h = draw.textsize(msg, font=font)
    draw.text((t5-w/3,t6/3-w/2), msg,(color[0],color[1],color[2]),font=font)
    #shape_img= shape_img.rotate(random.randint(0,360))
    shape_img.save('test.png', 'PNG')
    
    return hexColors

def createRectangleTarget():
    color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
    hexColors = ["rectangle",webcolors.rgb_to_hex(color)]
    
    W, H = random.randint(400,600), random.randint(400,600)
    shape_img = Image.new('RGBA', (W, H), (color[0], color[1], color[2], 0))
    
    draw = ImageDraw.Draw(shape_img)    
    draw.rectangle((0, 0, W, H), fill=(color[0], color[1], color[2]))
    font = ImageFont.truetype("Helvetica.ttf", random.randint(250,450))
    # draw.text((x, y),"Sample Text",(r,g,b))
    color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
    
    msg = random.choice(string.ascii_letters)
    hexColors.append(msg)
    hexColors.append(webcolors.rgb_to_hex(color))
    
    w, h = draw.textsize(msg, font=font)
    draw.text(((W-w)/2,(H-h)/2), msg,(color[0],color[1],color[2]),font=font)
    #shape_img= shape_img.rotate(random.randint(0,360))
    shape_img.save('test.png', 'PNG')
    
    return hexColors

def createSemiCircleTarget():
    color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
    hexColors = ["semicircle",webcolors.rgb_to_hex(color)]
    W, H = random.randint(400,600), random.randint(400,600)
    shape_img = Image.new('RGBA', (W, int(H/2)), (color[0], color[1], color[2], 0))
    
    draw = ImageDraw.Draw(shape_img)
    draw.ellipse((0, 0, W, H), fill=(color[0], color[1], color[2]))
    font = ImageFont.truetype("Helvetica.ttf", random.randint(200,300))
    # draw.text((x, y),"Sample Text",(r,g,b))
    color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
    
    msg = random.choice(string.ascii_letters)
    hexColors.append(msg)
    hexColors.append(webcolors.rgb_to_hex(color))
    
    w, h = draw.textsize(msg, font=font)
    draw.text(((W-w)/2,6), msg,(color[0],color[1],color[2]),font=font)
    #shape_img= shape_img.rotate(random.randint(0,360))
    shape_img.save('test.png', 'PNG')
    return hexColors

def createQuarterCircleTarget():
    color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
    hexColors = ["quartercircle",webcolors.rgb_to_hex(color)]
    W, H = random.randint(400,600), random.randint(400,600)
    shape_img = Image.new('RGBA', (int(W/2), int(H/2)), (color[0], color[1], color[2], 0))
    
    draw = ImageDraw.Draw(shape_img)
    draw.ellipse((0, 0, W, H), fill=(color[0], color[1], color[2]))
    font = ImageFont.truetype("Helvetica.ttf", int(max(W/2, H/2)) - 100)
    # draw.text((x, y),"Sample Text",(r,g,b))
    color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
    
    msg = random.choice(string.ascii_letters)
    hexColors.append(msg)
    hexColors.append(webcolors.rgb_to_hex(color))
    
    w, h = draw.textsize(msg, font=font)
    draw.text((200-w,200-h), msg,(color[0],color[1],color[2]),font=font)
    #shape_img= shape_img.rotate(random.randint(0,360))
    shape_img.save('test.png', 'PNG')
    return hexColors

def createCrossTarget():
    color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
    hexColors = ["cross",webcolors.rgb_to_hex(color)]
    
    W, H = random.randint(400,600), random.randint(400,600)
    shape_img = Image.new('RGBA', (W, H), (color[0], color[1], color[2], 0))
    
    draw = ImageDraw.Draw(shape_img)    
    draw.rectangle((5*W/16, 0, 11*W/16, H), fill=(color[0], color[1], color[2]))
    draw.rectangle((0, 5*H/16, W, 11*H/16), fill=(color[0], color[1], color[2]))
    font = ImageFont.truetype("Helvetica.ttf", random.randint(250,450))
    # draw.text((x, y),"Sample Text",(r,g,b))
    color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
    
    msg = random.choice(string.ascii_letters)
    hexColors.append(msg)
    hexColors.append(webcolors.rgb_to_hex(color))
    
    w, h = draw.textsize(msg, font=font)
    draw.text(((W-w)/2,(H-h)/2), msg,(color[0],color[1],color[2]),font=font)
    #shape_img= shape_img.rotate(random.randint(0,360))
    shape_img.save('test.png', 'PNG')
    
    return hexColors

def createPolygonTarget():
    numSides = random.randint(5,8)
    if numSides == 5:
        name = "pentagon"
    elif numSides == 6:
        name = "hexagon"
    elif numSides == 7:
        name = "heptagon"
    else:
        name = "octagon"
       
    W = random.randint(400,600)
    H = W
    xy = [ 
            ((math.cos(th) + 1) * W/2, 
             (math.sin(th) + 1) * W/2) 
            for th in [i * (2 * math.pi) / numSides for i in range(numSides)] 
    ]        
    
    color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
    hexColors = [name,webcolors.rgb_to_hex(color)]
    
    
    shape_img = Image.new('RGBA', (W, H), (color[0], color[1], color[2], 0))
    
    draw = ImageDraw.Draw(shape_img)    
    draw.polygon(xy, fill=(color[0], color[1], color[2]))
    
    font = ImageFont.truetype("Helvetica.ttf", random.randint(250,450))
    # draw.text((x, y),"Sample Text",(r,g,b))
    color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
    
    msg = random.choice(string.ascii_letters)
    hexColors.append(msg)
    hexColors.append(webcolors.rgb_to_hex(color))
    
    w, h = draw.textsize(msg, font=font)
    draw.text(((W-w)/2,(H-h)/2), msg,(color[0],color[1],color[2]),font=font)
    #shape_img= shape_img.rotate(random.randint(0,360))
    shape_img.save('test.png', 'PNG')
    
    return hexColors

def createStarTarget():
    color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
    hexColors = ["star",webcolors.rgb_to_hex(color)]
    
    W, H = 500, 500
    shape_img = Image.new('RGBA', (W, H), (color[0], color[1], color[2], 0))
    
    draw = ImageDraw.Draw(shape_img)
    draw.polygon(((0,180),(500,180),(250,360)), fill=(color))
    draw.polygon(((250,0),(95,475),(345,290)), fill=(color))
    draw.polygon(((250,0),(405,475),(155,290)), fill=(color))
    font = ImageFont.truetype("Helvetica.ttf", 200)
    # draw.text((x, y),"Sample Text",(r,g,b))
    color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
    
    msg = random.choice(string.ascii_letters)
    hexColors.append(msg)
    hexColors.append(webcolors.rgb_to_hex(color))
    
    w, h = draw.textsize(msg, font=font)
    draw.text(((W-w)/2,(H-h)/2), msg,(color[0],color[1],color[2]),font=font)
    #shape_img= shape_img.rotate(random.randint(0,360))
    shape_img.save('test.png', 'PNG')
    
    return hexColors
    
def addSimilarColors(target):
    #back_color = target_img[1][1][0:3]
    b_channel, g_channel, r_channel, alpha_channel = cv2.split(target)
    targetLAB = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    
    for i in range(targetLAB.shape[0]):
        for j in range(targetLAB.shape[1]):
            #if target[i][j][0] == back_color[0] and target[i][j][1] == back_color[1] and target[i][j][2] == back_color[2]:
            targetLAB[i][j][0] += random.randint(-30,-10)
            targetLAB[i][j][1] += random.randint(-5,5)
            targetLAB[i][j][2] += random.randint(-5,5)
            
    target = cv2.cvtColor(targetLAB, cv2.COLOR_LAB2BGR)
    
    b_channel, g_channel, r_channel = cv2.split(target)
    
    target = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    
    return target
    



createCircle = False
createRect = True
createTri = False
createSemiCircle = False
createQuarterCircle = False
createCross = False
createPolygon = False
createStar = False

# CHANGE THESE
createAllTargets = True
saveImg = False
saveFullImg = True
numTargetsPerImg = 1 # inner
numImgToMake = 1 # outer
sizeOfTarget = 100 # number of pixels to make target

for j in range(numImgToMake):
    if createAllTargets:
        if j == 0:
            createCircle = True
            createRect = False
            createTri = False
            createSemiCircle = False
            createQuarterCircle = False
            createCross = False
            createPolygon = False
            createStar = False
        elif j == (numImgToMake) / 8:
            createCircle = False
            createRect = True
            createTri = False
            createSemiCircle = False
            createQuarterCircle = False
            createCross = False
            createPolygon = False
            createStar = False
        elif j == 2*(numImgToMake) / 8:
            createCircle = False
            createRect = False
            createTri = True
            createSemiCircle = False
            createQuarterCircle = False
            createCross = False
            createPolygon = False
            createStar = False
        elif j == 3*(numImgToMake) / 8:
            createCircle = False
            createRect = False
            createTri = False
            createSemiCircle = True
            createQuarterCircle = False
            createCross = False
            createPolygon = False
            createStar = False
        elif j == 4*(numImgToMake) / 8:
            createCircle = False
            createRect = False
            createTri = False
            createSemiCircle = False
            createQuarterCircle = True
            createCross = False
            createPolygon = False
            createStar = False
        elif j == 5*(numImgToMake) / 8:
            createCircle = False
            createRect = False
            createTri = False
            createSemiCircle = False
            createQuarterCircle = False
            createCross = True
            createPolygon = False
            createStar = False
        elif j == 6*(numImgToMake) / 8:
            createCircle = False
            createRect = False
            createTri = False
            createSemiCircle = False
            createQuarterCircle = False
            createCross = False
            createPolygon = True
            createStar = False
        elif j == 7*(numImgToMake) / 8:
            createCircle = False
            createRect = False
            createTri = False
            createSemiCircle = False
            createQuarterCircle = False
            createCross = False
            createPolygon = True
            createStar = False
    base_img = cv2.imread("test google earth.jpg")
    for i in range(numTargetsPerImg):
        fileInfo = []
        if createCircle:
            fileInfo = createCircleTarget()
        elif createRect:
            fileInfo = createRectangleTarget()
        elif createTri:
            fileInfo = createTriangleTarget()
        elif createSemiCircle:
            fileInfo = createSemiCircleTarget()
        elif createQuarterCircle:
            fileInfo = createQuarterCircleTarget()
        elif createCross:
            fileInfo = createCrossTarget()
        elif createPolygon:
            fileInfo = createPolygonTarget()
        elif createStar:
            fileInfo = createStarTarget()
        
        target_img = cv2.imread("test.png", cv2.IMREAD_UNCHANGED)
        
        random_rot = random.randint(1,4)
        if random_rot == 1:
            target_img = target_img
        elif random_rot == 2:
            target_img = cv2.rotate(target_img, cv2.ROTATE_90_CLOCKWISE)
        elif random_rot == 3:
            target_img = cv2.rotate(target_img, cv2.ROTATE_180)
        elif random_rot == 4:
            target_img = cv2.rotate(target_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        w_target = target_img.shape[0]
        
        shrink_size = sizeOfTarget / w_target
        
        #shrink_size = random.randint(10,18)/100
        target_img = cv2.resize(target_img, (0,0), fx=shrink_size, fy=shrink_size,interpolation = cv2.INTER_AREA) 
        target_img = addSimilarColors(target_img)
        #target_img = sp_noise(target_img, 1)
        extra_area = 200 # area on the sides of the map for a buffer
        x = random.randint(extra_area,base_img.shape[0]-extra_area)
        y = random.randint(extra_area,base_img.shape[1]-extra_area)
        
        # adds target and stores the x,y coordinate locations
        right_x, bottom_y = addTarget(target_img, base_img, x, y)
        
        cropped_border = 15 # border around the cropped images
        
        #crops the image around the target
        cropped_img = base_img[x-cropped_border:x+right_x+cropped_border, y-cropped_border:y+bottom_y+cropped_border]
        
        if saveImg:
            if createCircle:   
                cv2.imwrite('Image Dataset/Close Ups/Circle/'+fileInfo[0]+"_"+fileInfo[1]+"_"+fileInfo[2]+"_"+fileInfo[3]+".png", cropped_img) 
            elif createRect:
                cv2.imwrite('Image Dataset/Close Ups/Rectangle/'+fileInfo[0]+"_"+fileInfo[1]+"_"+fileInfo[2]+"_"+fileInfo[3]+".png", cropped_img)
            elif createTri:
                cv2.imwrite('Image Dataset/Close Ups/Triangle/'+fileInfo[0]+"_"+fileInfo[1]+"_"+fileInfo[2]+"_"+fileInfo[3]+".png", cropped_img)
            elif createSemiCircle:
                cv2.imwrite('Image Dataset/Close Ups/Semicircle/'+fileInfo[0]+"_"+fileInfo[1]+"_"+fileInfo[2]+"_"+fileInfo[3]+".png", cropped_img)
            elif createQuarterCircle:
                cv2.imwrite('Image Dataset/Close Ups/Quartercircle/'+fileInfo[0]+"_"+fileInfo[1]+"_"+fileInfo[2]+"_"+fileInfo[3]+".png", cropped_img)
            elif createCross:
                cv2.imwrite('Image Dataset/Close Ups/Cross/'+fileInfo[0]+"_"+fileInfo[1]+"_"+fileInfo[2]+"_"+fileInfo[3]+".png", cropped_img)
            elif createPolygon:
                cv2.imwrite('Image Dataset/Close Ups/Polygon/'+fileInfo[0]+"_"+fileInfo[1]+"_"+fileInfo[2]+"_"+fileInfo[3]+".png", cropped_img)
            elif createStar:
                cv2.imwrite('Image Dataset/Close Ups/Star/'+fileInfo[0]+"_"+fileInfo[1]+"_"+fileInfo[2]+"_"+fileInfo[3]+".png", cropped_img)
        
        if saveFullImg:
           cv2.imwrite('C:/Users/acrog/Documents/Purdue/IEEE/MyDataset/Full Images/'+fileInfo[0]+"_"+fileInfo[1]+"_"+fileInfo[2]+"_"+fileInfo[3]+".jpg", base_img)
           
    if j % 100 == 0:
        print(j)
    
if not saveImg:    
    cv2.imshow('shape image',cropped_img)
    cv2.imshow('base image',base_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()