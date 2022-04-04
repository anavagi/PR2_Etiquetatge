__authors__ = ['1489845','1529079','1600715']
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
        #Operation to calculate N = F·C
        x_shape = X.shape #shape of the matrix
        F = x_shape[0] #first value of the shape list (lists)
        C = x_shape[1] #second value of the shape list (columns)
        N = F * C #Calculate N
        
        self.X = np.reshape(X, (N, 3)) #Reshape matrix 



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
            options['fitting'] = 'WCD'  #within class distance.

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
        if self.K <=0:
            return
        
        if self.options['km_init'].lower() == 'first': #The first K dots
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
            
            sorted_index=sorted(row_indexes)
            
            centroids = []
            
            for indexIT in range(self.K):
                centroids.append(self.X[sorted_index[indexIT]])

            self.centroids = np.array(centroids)
            
        elif self.options['km_init'].lower() == 'random': #K random dots
            self.centroids = np.random.rand(self.K, self.X.shape[1])

        elif self.options['km_init'].lower() == 'custom': #TBImplemented
            pass
        
        self.old_centroids = self.centroids
        #print("matrix X",np.unique(self.X, axis=0))

#YOOOOOOOOOOO
    def get_labels(self):
        """Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        print(self.X)


    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass

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
        pass

    def whitinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        return np.random.rand()

    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass

#YOOOOOOOOOOO
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
    # P_xshape = X.shape[0] #P,D
    # K_cshape = C.shape[0] #K,D
    # matriu_distancia = np.zeros((P_xshape,K_cshape))
    # for i in len(K_cshape):
    #     matriu_distancia=((X-C[index])**2).sum(axis=1)  

    K = C.shape[0] # k es numero de files o numero de pixels, es el mateix 
    distanciaCalculada = np.zeros((X.shape[0],K))
    acumulador=np.zeros(())
    for index in range(K): # recorrem tots els centroids, farem el calcul de la distancia euclidiana 
        # print((X-C[index])**2)  
        acumulador=((X-C[index])**2).sum(axis=1)  
        # print(acumulador)
        distanciaCalculada[:,[index]]= np.sqrt(np.reshape(acumulador,(acumulador.shape[0],1)))
        # print(distanciaCalculada.shape)
        # print(distanciaCalculada) 
    return distanciaCalculada



def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    return list(utils.colors)
