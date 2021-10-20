import pandas as pd  
import numpy as np

from Knn import KNN
from utils import *

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