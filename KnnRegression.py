from KnnRegressionBase import KNNRegressionBase
from utils import *


class KNNRegression(KNNRegressionBase):

    def __init__(self, k=3):
        super().__init__(k)

    def regression(self, neighbours):
        values = [self.y_train[neighbour[1]] for neighbour in neighbours]
        return np.mean(values)
