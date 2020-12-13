# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:15:35 2020

@author: JEEVAN
"""


from tkinter import * 
from tkinter import Tk
from tkinter import ttk
from PIL import ImageTk,Image
from embeddings_extraction import embeddings
from recognize_face import recognition
from dataset_collection import dataset
from model_train import train


root = Tk()  
root.title("CRIMINAL DETECTION")
root.geometry("1000x700") 
root.configure(bg='black')

new_face_button = Button(root,text = "NEW CRIMINAL FACE RECOGNITION",
                         command = lambda:[dataset(), embeddings(), train(), recognition()],
                         activeforeground = "#ffffff",
                         activebackground = "#168900",
                         padx=10,
                         pady=20,
                         bg="#39FF14",
                         highlightcolor="#ADFF9E",
                         font="Impact",
                         width=30,
                         height=1)  
new_face_button.pack(side = TOP) 
   
recognizer_button = Button(root,text = "EXISTING CRIMINAL FACE RECOGNITION",
                           command = recognition,
                           activeforeground = "#ffffff",
                           activebackground = "#168900",
                           padx=10,
                           pady=20,
                           bg="#39FF14",
                           highlightcolor="#ADFF9E",
                           font="Impact",
                           width=30,
                           height=1)  
recognizer_button.pack(side = TOP) 

def resize_image(event):
    new_width = event.width
    new_height = event.height
    image = copy_of_image.resize((new_width, new_height))
    photo = ImageTk.PhotoImage(image)
    label.config(image = photo)
    label.image = photo

image = Image.open('Criminal_wallpaper.jpeg')
copy_of_image = image.copy()
photo = ImageTk.PhotoImage(image)
label = ttk.Label(root, image = photo)
label.bind('<Configure>', resize_image)
label.pack(fill=BOTH, expand = YES)
    

root.mainloop()        


      
