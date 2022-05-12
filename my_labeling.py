__authors__ = ['1489845', '1529079', '1600715']
# Nota: 1489845 i 1529079 pertanyen al grup DL17 i 1600715 pertany al grup DJ08
__group__ = ['DL17', 'DJ08']

import numpy as np
import Kmeans
import KNN
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
#import cv2

######################
##Anàlisi Qualitatiu##
######################

def Retrieval_by_color(test_imgs, test_labels, labels):  # Erik
    found_IMGs = []
    #https://www.geeksforgeeks.org/find-common-values-between-two-numpy-arrays/
    for it, img in enumerate(test_imgs):
       if np.array_equal(np.sort(np.intersect1d(test_labels[it], labels)), np.sort(labels)):
           found_IMGs.append(img)

    return np.array(found_IMGs)


def Retrieval_by_shape(train_imgs, train_labels, label):  # Skarleth
    found_IMGs = []

    for it, img in enumerate(train_imgs):
        if label in train_labels[it]:
            found_IMGs.append(img)

    return np.array(found_IMGs)

######################
##Anàlisi Quantitatiu#
######################


def Kmeans_statistics(KMean, Kmax):  # Erik

    WCD_list, K_list, it_list = [], [], []

    for i in range(Kmax-1):  # -1 ya que k=1 no se da
        K_list.append(KMean.K)
        KMean.fit()
        it_list.append(KMean.num_iter)
        WCD_list.append(KMean.whitinClassDistance())
        KMean.K += 1

    #SRC: https://matplotlib.org/stable/tutorials/introductory/pyplot.html

    #Gráfica WCD
    plt.plot((K_list), (WCD_list))
    plt.ylabel('WCD')
    plt.show()
    #Gráfica IT
    plt.plot((K_list), (it_list))
    plt.ylabel('Iteraciones')
    plt.show()

    # print(WCD_list)
    # print(K_list)
    # print(it_list)


def Get_shape_accuracy(labels_knn, Ground_Truth):  # Skarleth
    eq = []
    eq = (labels_knn == Ground_Truth)
    average = np.mean(eq)
    percentage = average*100
    return percentage


if __name__ == '__main__':

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
        test_imgs, test_class_labels, test_color_labels = read_dataset(
            ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))


######################
###Funciones Utiles###
######################

def Iniciar_KNN(nConjunt=len(train_imgs)):
    return KNN.KNN(train_imgs[:nConjunt], train_class_labels[:nConjunt])


def Iniciar_KMeans(nConjunt, K=2):
    #K=2 por defecto ya que es el minimo
    return Kmeans.KMeans(test_imgs[nConjunt], K)


def Iniciar_KmeansLabels(K=2):
    #K=2 por defecto ya que es el minimo

    labels = []

    for test_img in test_imgs:
        tmp = Kmeans.KMeans(test_img)
        tmp.find_bestK(K)
        labels.append(Kmeans.get_colors(tmp.centroids))

    return labels

######################
#########TEST#########
######################


while(True):

    MSG = """Introdueix un nº en funció del test que vols fer:
        1. Retrieval by Color
        2. Retrieval by Shape
        3. Kmeans Statistics
        4. Get Shape Accuracy
        5. Millores [SIN ACABAR]
        6. Sortir
    """
    print(MSG)

    seleccio = input()

    if not seleccio:
        print("Error, selecciona una altre vegada")
        continue

    seleccio = int(seleccio)

    if seleccio > 7 or seleccio < 1 or not seleccio:
        print("Error, selecciona una altre vegada")
        continue

    elif seleccio == 1:
        print("Iniciant Retrieval by Color")

        print("Introdueix un valor per a K")
        K = int(input())

        print("Introdueix el nº de imatges que vols retornar")
        nIMGs = int(input())

        test_img_labels = Iniciar_KmeansLabels()
        labels = []  # Colors que volem buscar

        while(True):
            print("Introdueix un color que vols buscar dels següents: ")

            for clase in np.unique(test_img_labels):
                print(clase)
            inputClase = input()

            if inputClase in np.unique(test_img_labels):
                labels.append(inputClase)

                print("Introduir un altre color (1=SI, altre nº = NO)?")
                sino = int(input())
                if sino == 1:
                    continue
                else:
                    break

        Retrieval = Retrieval_by_color(test_imgs, test_img_labels, labels)

        visualize_retrieval(Retrieval, nIMGs)

        continue

    elif seleccio == 2:
        print("Iniciant Retrieval by Shape")
        RetrievalKNN = Iniciar_KNN()

        print("Introdueix un valor per a K")
        K = int(input())

        nClasses = RetrievalKNN.predict(train_imgs, K)

        print("Introdueix el nº de imatges que vols retornar")
        nIMGs = int(input())

        while(True):
            print("introdueix una classe que vols buscar de les següents: ")

            for clase in np.unique(nClasses):
                print(clase)

            inputClase = input()

            if inputClase in np.unique(nClasses):
                break

            print("Error, introdueix una altre vegada")

        Retrieval = Retrieval_by_shape(train_imgs, nClasses, inputClase)

        visualize_retrieval(Retrieval, nIMGs)

        continue

    elif seleccio == 3:
        print("Iniciant Kmeans Statistics")

        print("Introdueix un valor per a K max")
        Kmax = int(input())

        print("Introdueix un valor per al nº de imatges que vols analitzar ( Max: ", len(test_imgs), ")")
        nConjunt = int(input())-1  # Restem 1 ja que comença per 0

        ExempleStatistics = Iniciar_KMeans(nConjunt)

        Kmeans_statistics(ExempleStatistics, int(Kmax))

        print("Retornades grafiques per a Kmax=", Kmax)

        continue

    elif seleccio == 4:
        print("Iniciant Get Shape Accuracy")

        print("Introdueix el valor máxim de K que vols analitzar")
        KMax = int(input())

        print("Introdueix el nº d'imatges que vols analitzar ( max:", len(train_imgs), ")")
        nConjunt = int(input())

        KNNShape = Iniciar_KNN(nConjunt)

        for K in range(2, KMax+1):  # +1 ja que es FINS on hem de buscar la K
            labels = KNNShape.predict(test_imgs[:nConjunt], K)
            print("K = ", K, "amb precisió --> ", Get_shape_accuracy(labels, test_class_labels[:nConjunt]), "%")
        continue

    elif seleccio == 5:
        print("Iniciant apartat millores [SIN ACABAR]")
        
        print("Llindar:")
        kmeanLlindar = Iniciar_KMeans(850)
        
        Llindar=0
        KList=[]
        LlindarList=[]
        
        for i in range(11):
          kmeanLlindar.find_bestK(6,Llindar) 
          #print("BEST K amb llindar =",Llindar,"-->",kmeanLlindar.K)
          LlindarList.append(Llindar)
          KList.append(kmeanLlindar.K)
          Llindar+=10
        
        plt.plot((LlindarList), (KList))
        plt.ylabel('Llindar respecte de K')
        plt.show()
        
        continue

    elif seleccio == 6:
        print("Exit")
        break
