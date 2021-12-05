import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv
import os

# Fonction de génération d'histogrammes pour images avec précision variable
def gen(k,data,varName):
    if os.path.isfile(f'./imgGen/{varName}RGB{k}.npy')==True:
        print("Attention, le fichier que vous voulez générer existe déjà veuillez le supprimer avant d'en recréer un autre !")
        return 0
    dataRGB=[]
    dataName=[]

    #boucle qui permet pour chaque images de le transformer en un vecteur de (2^K)*3 dimensions
    for file in data:
        img=cv.imread(f'{dir}{varName}/{file}' ,cv.IMREAD_COLOR)
        red_histo = cv.calcHist([img], [0], None, [k], [0,256])
        green_histo = cv.calcHist([img], [1], None, [k], [0,256])
        blue_histo = cv.calcHist([img], [2], None, [k], [0,256])
        cv.normalize(green_histo,green_histo)
        cv.normalize(red_histo,red_histo)
        cv.normalize(blue_histo,blue_histo)

        hist=np.concatenate([red_histo,green_histo,blue_histo])
        hist=np.ndarray.flatten(hist)
        print(f'finition {file}')
        dataName.append(file)
        dataRGB.append(hist) 
    np.save(f'./imgGen/{varName}RGB{k}', dataRGB)
    np.save(f'./imgGen/{varName}Name',dataName)



