import numpy as np
from math import sqrt

def euclidean_distance(first_row, second_row):
    if len(first_row) != len(second_row):
        raise Exception("Two rows must have the same dimension!")

    total = 0
    for index in range(len(first_row)):
        diff_square = (first_row[index] - second_row[index]) ** 2
        total += diff_square
    return sqrt(total)


def k_fold_cross_validation_split(X, k=5):
    # takes numpy array as an argument
    splitted_data = []
    data_partition_num = len(X)/k

    for i in range(k):
        start = int(i*data_partition_num)
        end = int((i+1)*data_partition_num)

        test_set = X[start:end,:]
        train_set = np.concatenate((X[0:start,:],X[end:,:]), axis=0)
        
        splitted_data.append(np.array([train_set,test_set], dtype=object))
    
    return np.array(splitted_data)