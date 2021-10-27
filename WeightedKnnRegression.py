from KnnRegressionBase import KNNRegressionBase
from utils import *


class WeightedKNNRegression(KNNRegressionBase):

    def __init__(self, k=3):
        super().__init__(k)

    def regression(self, neighbours):
        values = np.array([self.y_train[neighbour[1]] for neighbour in neighbours])
        weights = np.array([(1 / neighbour[0]) if neighbour[0] != 0 else 1 for neighbour in neighbours])

        return ((values * weights).sum()) / weights.sum()
