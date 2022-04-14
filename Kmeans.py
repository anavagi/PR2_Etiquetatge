__authors__ = ['1489845', '1529079', '1600715']
__group__ = 'GrupZZ'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictºionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the smaple space is the length of
                    the last dimension
        """
        F = X.shape[0]  # First value of the matrix shape (lists)
        C = X.shape[1]  # Second value of the shape list (columns)

        #Operation to calculate N = F·C
        N = F * C  # Calculate N

        # Creating a new matrix full of floatsx64 (not equal if float63 is lower or bigger)
        X = np.array(X, dtype=np.float64)
        self.X = np.reshape(X, (N, X.shape[2]))  # Reshape matrix

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other prameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """

        #No use if K<=0 because we cannot initialize the centroids
        if self.K <= 0:
            return

        if self.options['km_init'].lower() == 'first':  # The first K dots
            #OLD version
            # centroids = [] #Using this list both like iterator and store the centroids to treat them
            # for dotX in self.X:
            #     repeated = False
            #     for dotC in centroids: #Checking if stored centroids are repeated
            #         if np.array_equal(dotC, dotX):
            #             repeated = True
            #             continue
            #     if not repeated:
            #         centroids.append(dotX)
            #     if len(centroids) == self.K:
            #         break
            # self.centroids = np.array(centroids[:self.K])zz

            #SOURCE: https://stackoverflow.com/questions/54140523/retain-order-when-taking-unique-rows-in-a-numpy-array
            row_indexes = np.unique(self.X, return_index=True, axis=0)[1]

            sorted_index = sorted(row_indexes)

            centroids = []

            for indexIT in range(self.K):
                centroids.append(self.X[sorted_index[indexIT]])

            self.centroids = np.array(centroids)

        elif self.options['km_init'].lower() == 'random':  # K random dots
            self.centroids = np.random.rand(self.K, self.X.shape[1])

        elif self.options['km_init'].lower() == 'custom':  # TBImplemented
            pass

        self.old_centroids = self.centroids

    def get_labels(self):
        """Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        #Calculate de distance between points of X and the centroid
        matrix_distance = distance(self.X, self.centroids)

        #Takes the closest centroid (argmin by columns) and asign it to the variable labels
        self.labels = matrix_distance.argmin(axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ################UTILIZAR CÓDIGO PROPIO#################
        #######################################################
        self.old_centroids = np.copy(self.centroids)
        for i in range(self.centroids.shape[0]):
            points = self.labels == i
            if points.any():
                self.centroids[i] = self.X[points].reshape(
                    -1, self.X.shape[1]).mean(axis=0)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #SOURCE: https://www.codingem.com/numpy-compare-arrays/
        return (self.centroids == self.old_centroids).all()

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self._init_centroids()
        self.get_labels()
        self.get_centroids()

        # # self.num_iter = 0
        # while not self.converges():

        #     #4. Comprova si convergeix, en cas de no fer-ho torna al primer pas"""
        #     if (self.num_iter < self.options['max_iter']):
        #        break
        #     self.get_lables()
        #     self.get_centroids()

        #     self.num_iter += 
        
        while self.num_iter < self.options['max_iter'] and not self.converges():
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1

    def whitinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """
        WCD = 0 

        for centroid in range(self.centroids.shape[0]):
            cluster = self.labels == centroid
            if cluster.any():
                x = self.X[cluster]
                Cx = self.centroids[centroid]
                WCD += (np.linalg.norm(x - Cx)) ** 2
        
        
        WCD /= self.X.shape[0]
        
        return WCD

    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        whitin_class_distance = 0 #Initialize variable

        #Loop in range from 2 (defined value) to max_K
        for i in range(2,max_K):

            self.K = i #Assig to the number of clusters i value
            self.fit() #Call fit() function
            
            old_whitin_class_distance = whitin_class_distance #Save the old value
            whitin_class_distance = self.whitinClassDistance() #Keep whitinClassDistance() return value

            #Loop for i bigger than the defined value
            if i > 2: 
                #Dec function 100 * (WDCk / WDCk-1)
                per_DECk = 100 * (whitin_class_distance / old_whitin_class_distance)
                result_DECk = (100 - per_DECk)

                threshold = 20 #Example 20%
                #If result if smaller than 20 asign the previous K value (i-1)
                if not result_DECk > threshold: 
                    self.K = i - 1
                    break 
                #if result_DECk is bigger K = max_K


def distance(X, C):
    """
    Calculates the distance between each pixcel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    #dist sera una matriu amb els valors PK
    P_xshape = X.shape[0]  # P,D
    K_cshape = C.shape[0]  # K,D
    matriu_distance = np.zeros((P_xshape, K_cshape))

    #We use an auxiliar array to add the values of the distances
    aux_matrix = np.zeros(())

    for i in range(K_cshape):  # Uses range because object don't have type int if we use len()
        distance = (X-C[i])**2  # Calculates distance
        # Sum by columns, add to aux_matrix
        aux_matrix = (distance).sum(axis=1)

        F_dim = aux_matrix.shape[0]  # Take the first value of the dimension

        # Reshape matrix according to F_dim
        m_aux_reshape = np.reshape(aux_matrix, (F_dim, 1))

        #Assign into i value the square-root of the aux matrix
        matriu_distance[:, [i]] = np.sqrt(m_aux_reshape)

    return matriu_distance


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
    """

    lables = []

    for probability in utils.get_color_prob(centroids):
        # We use as an index the first max value we found (SOURCE: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html)
        lables.append(utils.colors[np.argmax(probability)])
    return lables
