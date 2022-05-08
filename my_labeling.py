__authors__ = ['1489845', '1529079', '1600715']
# Nota: 1489845 i 1529079 pertanyen al grup DL17 i 1600715 pertany al grup DJ08
__group__ = ['DL17', 'DJ08']

import numpy as np
import Kmeans
import KNN
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
#import cv2

if __name__ == '__main__':

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
        test_imgs, test_class_labels, test_color_labels = read_dataset(
            ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))


######################
##Anàlisi Qualitatiu##
######################


def Retrieval_by_color(test_imgs, color_labels, Color):
    found_IMGs = []
    #https://www.geeksforgeeks.org/find-common-values-between-two-numpy-arrays/
    for it, img in enumerate(test_imgs):
       if np.array_equal(np.sort(np.intersect1d(color_labels[it], Color)), np.sort(Color)):
           found_IMGs.append(img)

    return np.array(found_IMGs)


def Retrival_by_shape():
    pass


def Retrival_combined():
    pass

######################
##Anàlisi Quantitatiu#
######################

# No pasamos la clase KMeans ya que la calculamos dentro de la función con el nº de indice
def Kmeans_statistics(nIMG, test_imgs, Kmax):

    KMean = Kmeans.KMeans(test_imgs[nIMG], 2) #2 por defecto ya que es el minimo

    time_list, K_list, it_list = [],[],[]

    for k in range(Kmax-1): #-1 ya que k=1 no se da
        K_list.append(KMean.K)
        KMean.fit()
        it_list.append(KMean.num_iter)
        time_list.append(KMean.whitinClassDistance())
        KMean.K += 1

    #SRC: https://matplotlib.org/stable/tutorials/introductory/pyplot.html
   
    #Gráfica WCD
    plt.bar((K_list), (time_list))
    plt.ylabel('WCD')
    plt.show()
    #Gráfica IT
    plt.bar((K_list), (it_list))
    plt.ylabel('Iteraciones')
    plt.show()

    # print(time_list)
    # print(K_list)
    # print(it_list)

def Get_shape_accuracy(labels_knn,Ground_Truth):
    eq=[]
    eq=(labels_knn==Ground_Truth)
    average=np.mean(eq)
    percentage=average*100
    return percentage

def Get_colors_accuracy():
    pass

######################
########Millores######
######################


def Find_bestK():
    pass



######################
#########TEST#########
######################


#TEST Kmeans_statistics index=5, text_imgs, kmax=5
Kmeans_statistics(5, test_imgs, 5)
