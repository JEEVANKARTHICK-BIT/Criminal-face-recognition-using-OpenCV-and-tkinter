# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 18:57:20 2020

@author: JEEVAN
"""


import cv2
import time
import pickle
import imutils
import numpy as np
from imutils.video import VideoStream

    
def recognition():
    
    print("Loading the Face Detecting model...")
    
    caffemodel = "DNN_module/res10_300x300_ssd_iter_140000.caffemodel"
    prototxt = "DNN_module/deploy.prototxt.txt"    
    detector = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    
    print("Loading the Face Recognizer and label Encodings...")
    
    embedder = cv2.dnn.readNetFromTorch("OpenFace_embedding_model/openface_nn4.small2.v1.t7") 
    recognizer = pickle.loads(open("serialized_files/SVM_recognizer.pickle", "rb").read())
    le = pickle.loads(open("serialized_files/label_encoder.pickle", "rb").read())
    
    print("Initializing Video Stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    
    # Looping frames from Video Stream
    while True:

    	frame = vs.read()
    	frame = imutils.resize(frame, width=600)
    	(h, w) = frame.shape[:2]
    
    	# Creating image blob
    	imageBlob = cv2.dnn.blobFromImage(
    		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
    		(104.0, 177.0, 123.0), swapRB=False, crop=False)
    
    	# Using detector to localize faces from the images
    	detector.setInput(imageBlob)
    	detections = detector.forward()
    
    	for i in range(0, detections.shape[2]):
    		# extracting confidence from the detections
    		confidence = detections[0, 0, i, 2]
    
    		# Eliminating worst detections
    		if confidence > 0.9:
    			# Calculating the x,y coordinates
    			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    			(startX, startY, endX, endY) = box.astype("int")
    
    			# Getting Region of interest
    			face = frame[startY:endY, startX:endX]
    			(fH, fW) = face.shape[:2]
    
    			# Checking for sufficient face width and face height
    			if fW < 20 or fH < 20:
    				continue
    
    			# Passing the blob to the Embedding model 
    			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
    				(96, 96), (0, 0, 0), swapRB=True, crop=False)
    			embedder.setInput(faceBlob)
    			vec = embedder.forward()
    
    			# Getting classified face name
    			preds = recognizer.predict_proba(vec)[0]
    			j = np.argmax(preds)
    			proba = preds[j]
    			name = le.classes_[j]
    
    			# Bounding box around the face
    			text = "{}: {:.2f}%".format(name, proba * 100)
    			y = startY - 10 if startY - 10 > 10 else startY + 10
    			cv2.rectangle(frame, (startX, startY), (endX, endY),
    				(0, 0, 255), 2)
    			cv2.putText(frame, text, (startX, y),
    				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    	cv2.imshow("Frame", frame)
    	key = cv2.waitKey(1) & 0xFF
    
    	# press 'q' to stop
    	if key == ord("q"):
    		break
    
    cv2.destroyAllWindows()
    vs.stop()

