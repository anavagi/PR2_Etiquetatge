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

def Retrieval_by_color(test_imgs, test_labels, labels): #Erik
    found_IMGs = []
    #https://www.geeksforgeeks.org/find-common-values-between-two-numpy-arrays/
    for it, img in enumerate(test_imgs):
       if np.array_equal(np.sort(np.intersect1d(test_labels[it], labels)), np.sort(labels)):
           found_IMGs.append(img)

    return np.array(found_IMGs)


def Retrieval_by_shape(train_imgs, train_labels, label): #Skarleth
    found_IMGs = []

    for it, img in enumerate(train_imgs):
        if label in train_labels[it]:
            found_IMGs.append(img)

    return np.array(found_IMGs)

######################
##Anàlisi Quantitatiu#
######################


def Kmeans_statistics(KMean, test_imgs, Kmax):  # Erik

    WCD_list, K_list, it_list = [], [], []

    for k in range(Kmax-1):  # -1 ya que k=1 no se da
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


def Get_colors_accuracy(nIMG, test_imgs, ground_truth):  # Ana
    list_with_colors = []  # List to save our colors

    #---KMEAN----
    #cambiar el nIMG no varia el resultado
    KMean = Kmeans.KMeans(test_imgs[nIMG], 2)  # return one object
    KMean.find_bestK(3)  # Si variamos el valor varia el resultado ¡
    list_with_colors.append(Kmeans.get_colors(KMean.centroids))


######################
###Funciones Utiles###
######################

def Iniciar_KNN():
    return KNN.KNN(train_imgs, train_class_labels)


def Iniciar_KMeans(nConjunt, K=2):
    #K=2 por defecto ya que es el minimo
    return Kmeans.KMeans(test_imgs[nConjunt], K)

######################
#########TEST#########
######################


while(True):

    MSG = """Introdueix un nº en funció del test que vols fer:
        1. Retrieval by Color [SIN ACABAR]
        2. Retrieval by Shape
        3. Kmeans Statistics
        4. Get Shape Accuracy [SIN ACABAR]
        5. Get Color Accuracy [SIN ACABAR]
        6. Millores [SIN ACABAR]
        7. Sortir
    """

    print(MSG)

    seleccio = input()

    if not seleccio:
        print("Error, selecciona una altre vegada")
        continue

    seleccio = int(seleccio)

    if seleccio > 8 or seleccio < 1 or not seleccio:
        print("Error, selecciona una altre vegada")
        continue

    elif seleccio == 1:
        print("Iniciant Retrieval by Color [SIN ACABAR]")
        
        print("Introdueix un valor per a K")
        K = int(input())
        
        Retrieval = Iniciar_KMeans(len(test_imgs),K)
        
        #busquem la llista de colors
        
        
        
        
        continue

    elif seleccio == 2:
        print("Iniciant Retrieval by Shape")
        RetrievalKNN = Iniciar_KNN()

        print("Introdueix un valor per a K")
        K = int(input())

        classes = RetrievalKNN.predict(train_imgs, K)

        print("Introdueix el nº de imatges que vols retornar")
        nIMGs = int(input())

        while(True):
            print("introdueix una classe que vols buscar de les següents: ")

            for clase in np.unique(classes):
                print(clase)

            inputClase = input()

            if inputClase in np.unique(classes):
                break

            print("Error, introdueix una altre vegada")

        retrieval = Retrieval_by_shape(train_imgs, classes, inputClase)

        visualize_retrieval(retrieval, nIMGs)

        print("Retornades:", nIMGs, "imatges, tornem al menú principal. ")

        continue

    elif seleccio == 3:
        print("Iniciant Kmeans Statistics")

        print("Introdueix un valor per a K max")
        Kmax = int(input())

        print("Introdueix un valor per al nº de imatges que vols analitzar")
        nConjunt = int(input())

        ExempleStatistics = Iniciar_KMeans(nConjunt)

        Kmeans_statistics(ExempleStatistics, test_imgs, int(Kmax))

        print("Retornades grafiques per a Kmax=",Kmax)

        continue

    elif seleccio == 4:
        print("Iniciant Get Shape Accuracy [SIN ACABAR]")
        continue

    elif seleccio == 5:
        print("Iniciant Get Color Accuracy [SIN ACABAR]")
        continue

    elif seleccio == 6:
        print("Iniciant apartat millores [SIN ACABAR]")
        continue

    elif seleccio == 7:
        print("Sortint")
        break
