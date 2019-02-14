#!/bin/python


import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
import os
import random
import pandas as pd
from subprocess import call






os.chdir('./train-jpg')

Archivos = os.listdir()
imagenesMostrar= random.sample(Archivos,6)

Labels = pd.read_csv('train_v2.csv')

os.mkdir('ImagenesRecortadas')

font = cv2.FONT_HERSHEY_SIMPLEX

def recortarGuardar(file):  
     img=cv2.imread(file)
     gray = cv2.resize(img, (156,156))
     imageName = file.split('.')[0]
     imageTag = Labels.loc[Labels['image_name'].isin([imageName])].iloc[0,1]
     gray = cv2.putText(gray, imageTag ,(10,80),font, 0.4,300,2)
     cv2.imwrite('ImagenesRecortadas/Recor'+ file +'.jpg',gray)
     return gray



ListaImagenes = [recortarGuardar(y) for y in imagenesMostrar]

vstack1 = np.vstack((ListaImagenes[0],ListaImagenes[1]))
vstack2 = np.vstack((ListaImagenes[2],ListaImagenes[3]))
vstack3 = np.vstack((ListaImagenes[4],ListaImagenes[5]))

hstack1 = np.hstack((vstack1,vstack2))
hstack2 = np.hstack((hstack1,vstack3))


cv2.imshow("frame1",hstack2) #display in windows 
cv2.waitKey(0) 


