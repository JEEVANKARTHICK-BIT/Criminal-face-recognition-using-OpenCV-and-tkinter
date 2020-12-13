# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 18:53:13 2020

@author: JEEVAN
"""


import pickle
from sklearn.svm import SVC
from tkinter import *
from tkinter import Tk
from tkinter import messagebox
from sklearn.preprocessing import LabelEncoder



    
def train():
    
    window = Tk()
    window.eval('tk::PlaceWindow %s center' % window.winfo_toplevel())
    window.withdraw()
    
    print("Loading Face embeddings...")
    data = pickle.loads(open("serialized_files/serialized_embeddings.pickle", "rb").read())
    
    print("Encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    
    print("Training the model...")
    recognizer = SVC(C=1.5, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)
    
    # saving SVM_recognizer pickle file
    f = open("serialized_files/SVM_recognizer.pickle", "wb")
    f.write(pickle.dumps(recognizer))
    f.close()
    
    # saving label_encoder pickle file
    f = open("serialized_files/label_encoder.pickle", "wb")
    f.write(pickle.dumps(le))
    f.close()

    messagebox.showinfo('Model Training','Training completed successfully')    
    
    window.deiconify()
    window.destroy()
    window.quit()
