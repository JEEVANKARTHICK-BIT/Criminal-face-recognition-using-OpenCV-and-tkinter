# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 18:38:25 2020

@author: JEEVAN
"""


import os
import cv2
import pickle
import imutils
import numpy as np
from tkinter import *
from tkinter import Tk
from tkinter import messagebox
from imutils import paths
    
def embeddings():

    window = Tk()
    window.eval('tk::PlaceWindow %s center' % window.winfo_toplevel())
    window.withdraw()
    
    # load serialized face detector
    caffemodel = "DNN_module/res10_300x300_ssd_iter_140000.caffemodel"
    prototxt = "DNN_module/deploy.prototxt.txt"    
    detector = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    
    print("Loading the OpenFace model...")
    embedder = cv2.dnn.readNetFromTorch("OpenFace_embedding_model/openface_nn4.small2.v1.t7")
    
    print("Embedding the Faces...")
    imagePaths = list(paths.list_images("datasets"))
    
    # initializing face embeddings and names in lists
    knownEmbeddings = []
    knownNames = []
    
    count = 0
    
    # Image path looping
    for (i, imagePath) in enumerate(imagePaths):
    	print("{} of {} - Extracted images".format(i+1, len(imagePaths)))
    	name = imagePath.split(os.path.sep)[-2]
    
    	# load the image - resizing it - getting the dimensions
    	image = cv2.imread(imagePath)
    	image = imutils.resize(image, width=600)
    	(h, w) = image.shape[:2]
    
    	# Creating image blob
    	imageBlob = cv2.dnn.blobFromImage(
    		cv2.resize(image, (300, 300)), 1.0, (300, 300),
    		(104.0, 177.0, 123.0), swapRB=False, crop=False)
    
    	# Using detector to localize faces from the images
    	detector.setInput(imageBlob)
    	detections = detector.forward()
    
    	# ensuring atleast one face was detected
    	if len(detections) > 0:
    		i = np.argmax(detections[0, 0, :, 2])
    		confidence = detections[0, 0, i, 2]
    
    		# Eliminating worst detections
    		if confidence > 0.9:
    			# Calculating the x,y coordinates
    			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    			(startX, startY, endX, endY) = box.astype("int")
    
    			# Getting Region of interest
    			face = image[startY:endY, startX:endX]
    			(fH, fW) = face.shape[:2]
    
    			# Checking for sufficient face width and face height
    			if fW < 20 or fH < 20:
    				continue
    
    			# Passing the blob to the Embedding model
    			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
    				(96, 96), (0, 0, 0), swapRB=True, crop=False)
    			embedder.setInput(faceBlob)
    			vec = embedder.forward()
    
    			# appending corresponding names and embeddings into the list
    			knownNames.append(name)
    			knownEmbeddings.append(vec.flatten())
    			count += 1
    
    # Creating serialized_embeddings pickle file
    print("[INFO] serializing {} encodings...".format(count))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open("serialized_files/serialized_embeddings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()

    messagebox.showinfo('Embeddings Extraction','Serialized Embeddings crated successfully')    
    
    window.deiconify()
    window.destroy()
    window.quit()
