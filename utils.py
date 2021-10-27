import sys
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

def accuracy_score(predicted, labels):
    predicted = np.array(predicted)
    labels = np.array(labels)   

    accuracy = (predicted == labels).sum() / len(predicted)
    #print(accuracy*100)
    return accuracy*100

def mean_absolute_error(predicted, labels):
    predicted = np.array(predicted)
    labels = np.array(labels) 
    
    mae = np.absolute(labels-predicted).sum() / len(predicted)
    return mae
def min_max_normalization(data):
    #max_bucket = [-1000000000 for i in range(len(data[0]))]
    
    # working on temp data
    temp_data = data.copy()


    length = len(temp_data[0])
    for i in range(length):
        col_data = temp_data[:,i]
        _max = np.max(col_data)
        _min = np.min(col_data)

        # applying min-max normalization
        col_data = (col_data-_min)/(_max-_min)
        
        # replace data
        temp_data[:,i] = col_data
    

    """
    # finding maximum
    for row in temp_data:
        for index in range(len(row)):
            column = row[index]
            if column > max_bucket[index]:
                max_bucket[index] = column

    # finding minimum
    min_bucket = max_bucket.copy()
    for row in temp_data:
        for index in range(len(row)):
            column = row[index]
            if column < min_bucket[index]:
                min_bucket[index] = column

    # normalization part
    for row_index, row in enumerate(temp_data):
        for col_index in range(len(row)):
            column = row[col_index]
            normalized_value = (column - min_bucket[col_index]) / (max_bucket[col_index] - min_bucket[col_index])
            temp_data[row_index][col_index] = normalized_value
    """
    return temp_data

def k_fold_cross_validation_split(X, k=5):
    # takes numpy array as an argument
    # shuffle
    np.random.shuffle(X)
    
    
    splitted_data = []
    data_partition_num = len(X)/k

    for i in range(k):
        start = int(i*data_partition_num)
        end = int((i+1)*data_partition_num)

        test_set = X[start:end,:]
        train_set = np.concatenate((X[0:start,:],X[end:,:]), axis=0)
        
        splitted_data.append(np.array([train_set,test_set], dtype=object))
    
    return np.array(splitted_data)