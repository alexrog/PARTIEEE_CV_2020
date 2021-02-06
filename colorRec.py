import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import webcolors as wc

def showImage(im,title):
    cv2.imshow(title,im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def bilFilter(image):
    filtered = cv2.bilateralFilter(image,10,10,30)
    for i in range(30):
        filtered = cv2.bilateralFilter(filtered,10,10,30)
    showImage(filtered,'filtered')
    return filtered

def makeKMeans(filtered):
    #kmeans for color reduction below
    Z = filtered.reshape((-1,3))
    Z = np.float32(Z)
    
    K = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center)
    res = center[label.flatten()]
    reduced = res.reshape((img.shape))
    
    showImage(reduced,'reduced')
    
    return reduced, label, center

def BGRtoRGB(center):
    for i in range(3):
        tempB = center[i][0]
        center[i][0] = center[i][2]
        center[i][2] = tempB
    return center

def makeGraph(center, counts, fileName):
    colors = []
        
    for i in range(3):
        print(f'Color {i + 1}: {center[i]} with frequency {counts[i]}')
        str1 = ''
        for j in range(3):
            if j == 0:
                str1 += 'r: '
                str1 += str(center[i][j]) + ' '
            elif j == 1:
                str1 += 'g: '
                str1 += str(center[i][j]) + ' '
            else:
                str1 += 'b: '
                str1 += str(center[i][j])
        colors.append(str1)
    
    fig = plt.figure()
    plt.bar(colors, counts, color=(center[0] / 256, center[1] / 256, center[2] / 256))
    plt.title(fileName)
    plt.xlabel('RGB Values')
    plt.ylabel('Frequency')
    #plt.xticks(counts, colors)
    plt.show()
    return

def compareHex(filename, colors):
    #first hex is shape color, second hex is letter color
    splitFile = filename.split("_")
    splitFile[1] = splitFile[1].strip("#")
    splitFile[3] = splitFile[3].strip("#")
    
    return
    
if __name__ == '__main__':
    path = "Image Dataset/Close Ups/Circle/"
    fileList = os.listdir(path)
    for i in fileList:
        img = cv2.imread("Image Dataset/Close Ups/Circle/" + i)
        m,n,d = img.shape
        
        orig = img.copy()
        filtered = bilFilter(img)
        
        reduced, label, center = makeKMeans(filtered)
        
        # swap BGR to RGB, add the hex value at i to an array.
        center = BGRtoRGB(center)
        
        _, counts = np.unique(label, return_counts = True)
        
        makeGraph(center, counts, i)
        
        