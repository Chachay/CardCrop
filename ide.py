import sys, os
import tkinter
import numpy as np
import cv2
from PIL import ImageTk, Image

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# import matplotlib.backends.tkgg as tkgg
# from matplotlib.backends.backend_agg import FigureCanvasAgg

#----Settings-----------
maxCards = 30
filename ='image/20180624_163627.jpg' 
filename ='image/20161020_085332.jpg' 
maxsize = (1024, 768)

detectTh = 100
#-----------------------
def morphology(src):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    return cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)

def preprocessor(src):
    # Gray Scale + Otsu
    dest = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, dest = cv2.threshold(dest, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU) 
    kernel = np.ones((5,5),np.uint8)
    dest = cv2.erode(dest, kernel, iterations = 1)
    return dest

def find_cards(src):
    _, contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ret = []
    for contour in contours:
        contour = cv2.convexHull(contour)
        a = cv2.contourArea(contour)
        if a > detectTh:
            epsilon = 0.04*cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,epsilon,True)

            if len(approx) < 4: continue
            ret.append((a, approx))
    
    ret.sort(key = lambda t: t[0], reverse = True)
    area, contours = zip(*ret)
    return area, contours 

def getColor(cm):
    ret = (int(cm[0]*255), int(cm[1]*255), int(cm[2]*255))
    return ret

def TransformM(src):
    points = list(map(lambda x: x[0], src))

    points.sort(key=lambda x: x[0])
    LEFT = sorted(points[:2], key=lambda x: x[1])
    RIGHT = sorted(points[2:], key=lambda x: x[1])
    NW = LEFT[0]
    NE = RIGHT[0] 
    SE = RIGHT[1]
    SW = LEFT[1]
    
    Width  = ((NE[0]-NW[0])**2+(NE[1]-NW[1])**2)**0.5
    Height = ((SW[0]-NW[0])**2+(SW[1]-NE[1])**2)**0.5
    Aspect = Width/Height

    if Width > Height:
        h = np.array([ [0, 0], [800, 0], [800,800/Aspect],[0, 800/Aspect] ], np.float32)
    else:
        h = np.array([ [0, 0], [800*Aspect, 0], [800*Aspect,800],[0, 800] ], np.float32)

    s = np.array([NW, NE, SE, SW], np.float32)
    transform = cv2.getPerspectiveTransform(s, h)
    return transform, (int(h[2][0]),int(h[2][1]))


class Application(tkinter.Frame):
    class RectObj(object):
        def __init__(self, canvas, points):
            pass
    def __init__(self, master=None, filename=None):
        tkinter.Frame.__init__(self, master)
        self.grid()
        self.createWidgets()

        self.Objects = []
        self.lastIndex = 0

        self.dict = {}

        if filename:
            self.filename = filename
            self.loadimage()

            self.processedImg = preprocessor(self.image)
            self.pThumb = cv2.resize(self.processedImg, maxsize, interpolation=cv2.INTER_AREA)
            self.photoImage2 = ImageTk.PhotoImage(Image.fromarray(self.pThumb))

            self.canvas2.config(height=self.pThumb.shape[0], width = self.pThumb.shape[1])
            self.canvas2.create_image(0, 0, anchor=tkinter.NW, image=self.photoImage2)

            area, contours = find_cards(self.processedImg)

            plt.hist(area, bins=30)
            plt.show(block=False)

            TH = area[0]*0.5 if len(area) < 3 else area[2]*0.5

            self.displayimage()
            for i, contour in enumerate(contours[0:30]):
                if area[i] > TH:
                    if len(contour)==4:
                        transform, shape = TransformM(contour)
                        warp = cv2.warpPerspective(self.image, transform,shape)
                        cv2.imwrite("processed/%i.jpg"%i, warp)
                        self.createSQ(contour)

    def createWidgets(self):
        self.canvas = tkinter.Canvas(self, height = 1, width = 1, relief = tkinter.SUNKEN)
        self.canvas.grid(row = 0, column = 0)

        self.canvas2 = tkinter.Canvas(self, height = 1, width = 1, relief = tkinter.SUNKEN)
        self.canvas2.grid(row = 0, column = 1)

        self.canvas3 = tkinter.Canvas(self, height = 1,  width = 1, relief = tkinter.SUNKEN)
        self.canvas3.grid(row = 1, column = 0)
    
    def createSQ(self, points):
        color = "#%02x%02x%02x" % getColor(cm.jet((self.lastIndex%10)/10.))

        _points =list(map(lambda x: (int(x[0][0]/self.scale[0]), int(x[0][1]/self.scale[1]) ), points))
        rectID = self.canvas.create_polygon(*_points, fill=color, stipple = "gray25", outline="black")
        self.canvas.tag_bind(rectID, '<ButtonRelease-3>', self.canvas_mouse3up_callback)    

        for i, (x, y) in enumerate(_points):
            objId = self.canvas.create_oval((x-4, y-4, x+4, y+4), fill=color)
            self.dict[objId] = (i, rectID)
            self.canvas.tag_bind(objId, '<Button-1>', self.canvas_mouse1down_callback)    
            self.canvas.tag_bind(objId, '<B1-Motion>', self.canvas_mouse1move_callback)    
            self.canvas.tag_bind(objId, '<ButtonRelease-1>', self.canvas_mouse1up_callback)    

        self.Objects.append((self.lastIndex, points, color, rectID))
        self.lastIndex = self.lastIndex + 1

    def loadimage(self):
        self.image = cv2.imread(self.filename)
        self.image_rect = Rect(self.image.shape[:2])
    
    def displayimage(self):
        self.image_thumb = cv2.resize(self.image, maxsize, interpolation=cv2.INTER_AREA)
        self.image_thumb_rect = Rect(self.image_thumb.shape[:2])

        x_scale = float(self.image_rect.w)/self.image_thumb_rect.w
        y_scale = float(self.image_rect.h)/self.image_thumb_rect.h
        self.scale=(x_scale, y_scale)

        self.ImageBuffer = Image.fromarray(self.image_thumb[:,:,::-1])
        self.photoimage = ImageTk.PhotoImage(self.ImageBuffer)
        self.canvas.config(width=self.image_thumb_rect.w, height=self.image_thumb_rect.h)
        self.canvas.create_image(0, 0, anchor=tkinter.NW, image=self.photoimage)

    def canvas_mouse1down_callback(self, event):
        self.objID = self.canvas.find_closest(event.x, event.y)[0]
        self.objPos = (event.x, event.y)
        print(self.objID)

    def canvas_mouse1move_callback(self, event):
        if self.objID == None: return
        index, rectID = self.dict[self.objID]
        x0, y0, x1, y1 = self.canvas.coords(self.objID)
        points = self.canvas.coords(rectID)
        print(points)

        deltaX =  event.x - self.objPos[0]
        deltaY =  event.y - self.objPos[1]
        self.canvas.coords(self.objID, 
                            x0 + deltaX,
                            y0 + deltaY,
                            x1 + deltaX,
                            y1 + deltaY)
        points[index*2] = points[index*2] + deltaX
        points[index*2 + 1] = points[index*2 + 1] + deltaY
        self.canvas.coords(rectID, *points)

        self.objPos = (event.x, event.y)

    def canvas_mouse1up_callback(self, event):
        self.objID = None

    def canvas_mouse3up_callback(self, event):
        objID = self.canvas.find_closest(event.x, event.y)[0]

        keys = []
        for key, value in self.dict.items():
            if value[1] != objID:
                continue
            self.canvas.delete(key)
            keys.append(key)
        for key in keys:
            del(self.dict[key])

        self.canvas.delete(objID)
        for i, item in enumerate(self.Objects):
            print(item)
            if item[3] == objID:
                del(self.Objects[i])
                print(i, objID,item)

class Rect(object):
    def __init__(self, *args):
        self.set_points(*args)
    
    def set_points(self, *args):
        if len(args) == 1:
            pt1 = (0, 0)
            pt2 = args[0]
        else:
            pt1 = (0, 0)
            pt2 = (0, 0)

        y1, x1 = pt1
        y2, x2 = pt2

        self.left = min(x1, x2)
        self.right = max(x1, x2)
        self.top = min(y1, y2)
        self.bottom = max(y1, y2)
        self._update_dims()
    
    def _update_dims(self):
        self.w = self.right - self.left
        self.h = self.bottom - self.top

    def scale_rect(self, scale):
        x_scale = scale[0]
        y_scale = scale[1]
        r = Rect()
        r.top = int(self.top * y_scale)
        r.bottom = int(self.bottom * y_scale)
        r.right  = int(self.right * x_scale)
        r.left   = int(self.left * x_scale)
        r._update_dims()
        return r

def main():
    app = Application(filename = filename)
    app.master.title('Photo Cropper')
    app.mainloop()

if __name__=='__main__': main()