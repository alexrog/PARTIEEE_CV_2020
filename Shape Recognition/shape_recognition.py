import cv2
import numpy as np
from sklearn.cluster import KMeans


# see https://stackoverflow.com/questions/29156091/opencv-edge-border-detection-based-on-color

class ShapeRecognition:
    # lower sigma is tighter https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    def __auto_canny(self, image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        # return the edged image
        return edged

    def __classify_contour(self, image, contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)

        print(len(approx))
        cv2.drawContours(image, [approx], 0, (255, 0, 0), 2)


        # classify based on the number of vertices
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            shape = "rectangle"
        elif len(approx) == 7:
            shape = "heptagon"
        elif len(approx) == 10:
            shape = "star"
        elif len(approx) == 12:
            shape = "cross"
        else:
            shape = "circle"

        # return the name of the shape
        return shape

    def __select_contour(self, image):
        contours = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

        # get rid of contours that touch the border
        # for contour in contours:
        #     rect = cv2.boundingRect(contour)
        #     if rect[0] <= 0 or rect[1] <= 0 or rect[2] >= image.shape[1] or rect[3] >= image.shape[0]:
        #         contours.remove(contour)
        # if len(contours) == 0:
        #     return None

        return max(contours, key=cv2.contourArea)

    def __preprocess_kmeans(self, image, k):
        imageArr = image.reshape((image.shape[0] * image.shape[1], 3))
        clt = KMeans(n_clusters=k)
        clt.fit(imageArr)
        colors = clt.cluster_centers_
        for x in range(0, colors.shape[0]):
            for y in range(0, colors.shape[1]):
                colors[x][y] = round(colors[x][y])

        colors = colors.astype('uint8')

        imageArr = colors[clt.predict(imageArr)]
        image = imageArr.reshape((-1, image.shape[1], 3))

        return image


    def __preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        h = (h + 25) % 180
        image = h
        high_thresh, thresh_im = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = cv2.Canny(image, high_thresh / 2, high_thresh)

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.GaussianBlur(image, (5, 5), 0)

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # image = clahe.apply(image)

        # high_thresh, thresh_im = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # image = cv2.Canny(image, high_thresh / 1.5, high_thresh)
        # image = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY)[1]

        return image

    def classify_image(self, image):
        cv2.imshow("Target", image)
        cv2.waitKey()

        imageK = self.__preprocess_kmeans(image, 3)

        imageP = self.__preprocess_image(imageK)
        contour = self.__select_contour(imageP)
        if contour is None:
            return None

        cv2.drawContours(imageK, [contour], 0, (255, 0, 0), 2)
        shape = self.__classify_contour(imageP, contour)
        print(shape)

        cv2.imshow("Target", imageP)
        cv2.waitKey()

        cv2.imshow("Target", imageK)
        cv2.waitKey()
