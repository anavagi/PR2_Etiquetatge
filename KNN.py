__authors__ = ['1489845', '1529079', '1600715']
# Nota: 1489845 i 1529079 pertanyen al grup DL17 i 1600715 pertany al grup DJ08
__group__ = ['DL17', 'DJ08']

from cgi import test
import numpy as np
import math
import operator
from scipy.spatial.distance import cdist

class KNN:
    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################


    def _init_train(self,train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        # train_data = np.asarray(train_data, dtype=np.float64)
        # self.train_data = np.reshape(train_data, (train_data.shape[0],4800*3))


    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        # 1. Change dimensions (10, 60, 80, 3) Matrix(NxK)
        # print(test_data.shape)
        N = test_data.shape[0]
        K = test_data.shape[1] * test_data.shape[2] * test_data.shape[3] #Calculate K value

        test_data = np.array(test_data, dtype=np.float64) #Create array, type floats
        test_data = np.reshape(test_data, (N, K))  # Reshape matrix

        # 2. Calculate distance
        distances = cdist(test_data, self.train_data, 'euclidean')

        # 3. Save self.neighbors
        values = np.array() #auxiliar array that will become into a np.array
        for distance in distances:
            # print("k value",k)
            ordered_val = distance.argsort()[:k] #Values [first value: value k]
            # print(ordered_val)
            values.append(self.labels[ordered_val]) #Append values into array
        
        np.concatenate((self.neighbors, values), axis=0) #Turn list into array



    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        return np.random.randint(10, size=self.neighbors.size), np.random.random(self.neighbors.size)


    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """


        return np.random.randint(10, size=self.neighbors.size), np.random.random(self.neighbors.size)
