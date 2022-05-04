__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import numpy as np
import Kmeans
import KNN
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
#import cv2

if __name__ == '__main__':

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))



## You can start coding your functions here

def Retrieval_by_color(test_imgs,color_labels,Color): 
    found_IMGs = []
    #https://www.geeksforgeeks.org/find-common-values-between-two-numpy-arrays/    
    for it,img in enumerate(test_imgs):
       if np.array_equal(np.sort(np.intersect1d(color_labels[it], Color)), np.sort(Color)):
           found_IMGs.append(img)
          
    return np.array(found_IMGs)
