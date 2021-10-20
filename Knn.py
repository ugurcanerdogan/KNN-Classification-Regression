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
            [dist, index_of_neighbour],
            [another_dist, index_of_another_neighbour],
            ...
        ]
        """

        # distance calculation
        distances = [[euclidean_distance(x, self.X_train[i]), i] for i in range(len(self.X_train))]

        # sorting to find nearest neighbour
        print(distances)
        sorted_array = sorted(distances, key=lambda x :x[0])
        print(sorted_array)

        print("Asked data: ", x.astype(int))
        print("Nearest Neighbours:")
        for m in range(self.k):
            index = sorted_array[m][1]
            print(self.X_train[index], ", class: " ,self.y_train[index])