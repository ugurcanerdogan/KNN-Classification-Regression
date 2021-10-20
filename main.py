import pandas as pd  
import numpy as np

from utils import *



class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        pass
    
    def predict_helper(self, x):
        """
        [
            [dist, index_of_neihgbour], 
            [another_dist, index_of_another_neighbour], 
            ...
        ]
        """

        # distance calculation
        distances = [[euclidean_distance(x, X_train[i]), i] for i in range(len(self.X_train))]

        # sorting to find nearest neighbour
        print(distances)
        sorted_array = sorted(distances, key=lambda x:x[0])
        print(sorted_array)

        print("Asked data: ", x.astype(int))
        print("Nearest Neighbours:")
        for m in range(3):
            index = sorted_array[m][1]
            print(self.X_train[index], ", class: ",self.y_train[index])



data = pd.read_csv("glass.csv")

data = np.array(data)
splitted = k_fold_cross_validation_split(data, 5)
print(splitted.shape)

sample_train = splitted[0][0]
sample_test = splitted[0][1]
print(sample_train.shape)
print(sample_test.shape)

X_train = sample_train[:,:-1]
y_train = sample_train[:,-1]

X_test = sample_test[:,:-1]
y_test = sample_test[:,-1]


knn = KNN()
knn.fit(X_train, y_train)
knn.predict_helper(X_test[0])

print(y_test[0])