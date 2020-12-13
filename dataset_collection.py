# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 19:13:28 2020

@author: JEEVAN
"""

import os
import cv2
import time
import imutils
import numpy as np
from tkinter import *
from tkinter import Tk
from tkinter import Entry
from tkinter import messagebox
from imutils.video import VideoStream

  
def dataset():

    window = Tk()
    window.eval('tk::PlaceWindow %s center' % window.winfo_toplevel())
    window.withdraw()

    messagebox.showinfo('Dataset Collection','### Enter Criminal name in command line ###')
    
    face_name = input("Enter criminal name: ")

    caffemodel = "DNN_module/res10_300x300_ssd_iter_140000.caffemodel"
    prototxt = "DNN_module/deploy.prototxt.txt"
    
    print("Capturing face for dataset collection... \nLook at the camera...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    
    dataset = "datasets"
    name = face_name
    
    path = os.path.join(dataset,name)
    if not os.path.isdir(path):
        os.makedirs(path)
        
    (width,height) = (400,400)
    
    
    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    
    count = 1
    
    while (count <= 50):
        print("image ", count, " collected")
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
    
        for i in range(0, detections.shape[2]):
    
            confidence = detections[0, 0, i, 2]
    
            if confidence < 0.8:
                continue
    
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            onlyface = frame[startY-40:endY+40,startX-40:endX+40]
            resizeImg = cv2.resize(onlyface,(width,height))
            cv2.imwrite("%s/%s.jpg"%(path,count),resizeImg)
            count+=1
    
        cv2.imshow("Dataset Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break    
        
    messagebox.showinfo('Dataset Collection','Dataset collected successfully')
    
    window.deiconify()
    window.destroy()
    window.quit()
    
    cv2.destroyAllWindows()
    vs.stop()
