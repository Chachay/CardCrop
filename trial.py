import numpy as np
import cv2

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import matplotlib.backends.tkgg as tkgg
from matplotlib.backends.backend_agg import FigureCanvasAgg

def contrast(img):
    ret= np.clip((img - np.mean(img))/np.std(img)*3 + 125, 0, 255) 
    return cv2.convertScaleAbs(ret)

def getColor(cm):
    ret = (cm[0]*255, cm[1]*255, cm[2]*255)
    return ret

def findRect(points):
    points = list(map(lambda x: x[0], points))
    points = sorted(points, key= lambda x: x[1])

    neiborhood4 = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]],
                            np.uint8)
 
def morphology(src):
    neiborhood8 = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
                            np.uint8)
    return cv2.morphologyEx(src, cv2.MORPH_CLOSE, neiborhood8)

img = cv2.imread('image/20161020_085332.jpg')
img = cv2.resize(img,(1000,640))
img_size = img.shape[0]*img.shape[1]
max_cards = 30
detectTh  = img_size/max_cards*0.2
print(detectTh)

preprocess = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#preprocess = contrast(preprocess)
# _, preprocess = cv2.threshold(preprocess,125,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
preprocess = cv2.adaptiveThreshold(preprocess,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,221,0)

# preprocess = morphology(preprocess)

#B_preprocess = contrast(img[:,:,0])
B_preprocess = img[:,:,0]
_, B = cv2.threshold(B_preprocess, 125, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
cv2.imshow('sample', cv2.resize(B,(1000,640)))
cv2.waitKey(0)

# B_preprocess = contrast(img[:,:,1])
# _, G = cv2.threshold(B_preprocess, 125, 255, cv2.THRESH_BINARY)

# B_preprocess = contrast(img[:,:,2])
# _, R = cv2.threshold(B_preprocess, 125, 255, cv2.THRESH_BINARY)

# _, preprocess = cv2.threshold(preprocess, 170, 255, cv2.THRESH_TOZERO_INV )
# preprocess = cv2.bitwise_not(preprocess)
# _, thresh1 = cv2.threshold(preprocess,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
# preprocess = preprocess + B
cv2.imshow('sample', cv2.resize(preprocess,(1000,640)))
cv2.waitKey(0)

image, contours, hierarchy = cv2.findContours(preprocess,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
area = []
i = 0

for contour in contours:
    a = cv2.contourArea(contour)
    if a > detectTh:
        area.append(a)
        epsilon = 0.05*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,epsilon,True)
        rect = cv2.minAreaRect(approx)
        box  = cv2.boxPoints(rect)
        box  = np.int0(box)
        if approx.size < 10:
            # img = cv2.drawContours(img, [contour], -1, getColor(cm.jet((i%10)/10.)), 3)
            img = cv2.drawContours(img, [approx], -1, getColor(cm.jet((i%10)/10.)), 3)
        # img = cv2.drawContours(img, [box], -1, getColor(cm.jet((i%10)/10.)), 3)
        i = i + 1

print(len(area))
plt.hist(area)
plt.show()

cv2.imshow('sample', cv2.resize(img,(1000,640)))
cv2.waitKey(0)
