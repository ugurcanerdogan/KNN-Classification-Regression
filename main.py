import pandas as pd  
import numpy as np

from Knn import KNN
from WeightedKnn import WeightedKNN
from utils import *
from tqdm import tqdm

#X_train_normalized = min_max_normalization(X_train)


def cross_validation(splitted_data, knn):
    accuracies = []
    for data in tqdm(splitted_data):
        sample_train = data[0]
        sample_test  = data[1]

        #print(sample_train.shape)
        #print(sample_test.shape)

        # train and test sets
        X_train = sample_train[:,:-1]
        y_train = sample_train[:,-1]

        X_test = sample_test[:,:-1]
        y_test = sample_test[:,-1]
        
        # fitting data
        knn.fit(X_train, y_train)

        predictions = knn.predict(X_test)

        #print(knn.y_train)
        #print(predictions)
        #print(y_test)

        
        # calculate accuracy
        accuracy = accuracy_score(predictions, y_test)
        accuracies.append(accuracy)
    
    accuracies = np.array(accuracies)
    #print(accuracies)
    return np.sort(accuracies), np.mean(accuracies)

def main():


    data = pd.read_csv("glass.csv")

    data = np.array(data)
    splitted = k_fold_cross_validation_split(data, 5)
    #print(splitted.shape)

    #knn = KNN(k=7)
    knn = WeightedKNN(k=7)

    accuracies, mean_acc = cross_validation(splitted, knn)
    #print(mean_acc)

    print(mean_acc)



if __name__ == "__main__":
    main()