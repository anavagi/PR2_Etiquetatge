__authors__ = ['1489845', '1529079', '1600715']
# Nota: 1489845 i 1529079 pertanyen al grup DL17 i 1600715 pertany al grup DJ08
__group__ = ['DL17', 'DJ08']

import numpy as np
import Kmeans
import KNN
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import utils
from utils import *
import matplotlib.pyplot as plt
#import cv2

if __name__ == '__main__':

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
        test_imgs, test_class_labels, test_color_labels = read_dataset(
            ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

from my_labeling import *

def seleccioApartat():
    apartat = 0
    
    while(True): #Selecció d'apartat
        print("Selecciona l'apartat que vols revisar")
        print("    1 - Anàlisi qualitatiu")
        print("    2 - Anàlisi quantitatiu")
        print("    3 - Millores als mètodes de classificació")
        
        apartat = input()
        
        if int(apartat)<1 or int(apartat)>3 or not apartat:
            print("Error, selecciona una altre vegada")
        else:
            break
    
    return int(apartat)

def testQualitatiu():
    num = 0
    
    while(True): #Selecció de test
        print("Selecciona el test que vols realitzar")
        print("    1 - Test Retrieval_by_color")
        print("    2 - Test Retrieval_by_shape")
        
        num = input()
        
        if int(num)<1 or int(num)>2 or not num:
            print("Error, selecciona una altre vegada")
        else:
            break
    
    
    if int(num)==1:
        pass
    
    if int(num)==2:
        pass


def testQuantitatiu():
    num = 0
    
    while(True): #Selecció de test
        print("Selecciona el test que vols realitzar")
        print("    1 - Test Kmean_statistics")
        print("    2 - Test Get_shape_accuracy")
        print("    3 - Test Get_color_accuracy")
        
        num = input()
        
        if int(num)<1 or int(num)>3 or not num:
            print("Error, selecciona una altre vegada")
        else:
            break
    
    if int(num)==1:
        print("Introdueix un valor de K per K")
        K = input()
        print("Introdueix un valor del nº de imatges")
        n_imgs = input()
        
        Kmeans_statistics(int(n_imgs), test_imgs,int(K))
        
    elif int(num)==2:
        pass

    elif int(num)==3:
        pass
    
def testMillores():
    pass

def initTest():
    apartat = seleccioApartat()
    
    if apartat==1:
        testQualitatiu()
        
    elif apartat==2:
        testQuantitatiu()
        
    elif apartat==3:
        testMillores()
    
initTest()