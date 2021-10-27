from numpy.core.numeric import cross
import pandas as pd  
import numpy as np

from Knn import KNN
from WeightedKnn import WeightedKNN
from KnnRegression import KNNREGRESSION
from utils import *
from tqdm import tqdm

#X_train_normalized = min_max_normalization(X_train)


def cross_validation(splitted_data, knn, normalize, classification):
    accuracies = []
    mae_values = []
    result = None
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
        
        if normalize:
            """
                Applying min-max normalization to data
                
                Applying separately to avoid data leakage
            """
            X_train = min_max_normalization(X_train)
            X_test = min_max_normalization(X_test)

        # fitting data
        knn.fit(X_train, y_train)

        # prediction part
        predictions = knn.predict(X_test)

        #print(knn.y_train)
        #print(predictions)
        #print(y_test)


        if classification:
            # calculate accuracy
            accuracy = accuracy_score(predictions, y_test)
            accuracies.append(accuracy) 
        else:
            mae = mean_absolute_error(predictions, y_test)
            mae_values.append(mae)
            

    if classification:
        accuracies = np.array(accuracies)
        return np.sort(accuracies), np.mean(accuracies)
    else:
        return np.sort(mae_values), np.mean(mae_values)
    
    

def main():
    # defining random seed to get same random results every time
    # np.random.seed(12345)


    # classification part
    data = pd.read_csv("glass.csv")

    data = np.array(data)

    # 5 fold cross validation
    splitted = k_fold_cross_validation_split(data, 5)
    #print(splitted.shape)

    # k = 7
    knn1 = KNN(k=7)
    knn = WeightedKNN(k=7)

    accuracies, mean_acc = cross_validation(splitted, knn, normalize=True, classification=True)
    accuracies1, mean_acc1 = cross_validation(splitted, knn1, normalize=True, classification=True)
    #print(mean_acc)
    print("Distance based: ",mean_acc1, accuracies1)
    print("Weighted: ",mean_acc, accuracies)


    # regression part
    regression_data = pd.read_csv("Concrete_Data_Yeh.csv")
    regression_data = np.array(regression_data)

    knn2 = KNNREGRESSION(k=7)
    regression_splitted = k_fold_cross_validation_split(regression_data, 5)
    mae_values, mean_mae = cross_validation(regression_splitted, knn2, normalize=True, classification=False)

    print("Mae values: ",mae_values)
    print("Mean mae: ", mean_mae)
    accuracies






if __name__ == "__main__":
    main()