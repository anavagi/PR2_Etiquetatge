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

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """

        # numpy arrays are float by default. 14400 'cause --> 60x80x3 (RGB)
        #SRC: https://numpy.org/doc/stable/reference/generated/numpy.var.html#:~:text=For%20arrays%20of%20integer%20type,same%20as%20the%20array%20type.
        self.train_data = np.reshape(np.array(train_data), ((len(train_data)), 14400))

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
        K = test_data.shape[1] * test_data.shape[2] * test_data.shape[3]

        # Create array, type floats
        test_data = np.array(test_data, dtype=np.float64)
        test_data = np.reshape(test_data, (N, K))  # Reshape matrix

        # 2. Calculate distance
        distances = cdist(test_data, self.train_data, 'euclidean')

        # 3. Save self.neighbors
        values = []  # auxiliar array that will become into a np.array
        for distance in distances:
            # print("k value",k)
            # Values [first value: value k]
            ordered_val = distance.argsort()[:k]
            # print(ordered_val)
            values.append(self.labels[ordered_val])  # Append values into array

        self.neighbors = np.array(values)  # Turn list into array

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        maxNeighbors = np.array([])
        for i in self.neighbors:  # For each neightbor we search for the one with highest value
            value, count = np.unique(i, return_counts=True)  # /!\ 
            maxNeighbors = np.append(maxNeighbors, value[np.argmax(count)])   # Store the highest value
        return maxNeighbors

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
