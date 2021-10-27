import pandas as pd

from Knn import KNN
from KnnRegression import KNNRegression
from WeightedKnn import WeightedKNN
from WeightedKnnRegression import WeightedKNNRegression
from utils import *

def generalTest():
    # classification part
    classification_data = pd.read_csv("glass.csv")
    classification_data = np.array(classification_data)

    # 5 fold cross validation
    splitted = k_fold_cross_validation_split(classification_data, 5)

    # regression part
    regression_data = pd.read_csv("Concrete_Data_Yeh.csv")
    regression_data = np.array(regression_data)

    regression_splitted = k_fold_cross_validation_split(regression_data, 5)

    for i in range(1, 6):
        k_val = (2 * i) - 1

        for bool in [True, False]:
            text = "With" if bool else "Without"
            print(f"{text} Normalization")
            knn = KNN(k=k_val)
            knn1 = WeightedKNN(k=k_val)

            accuracies, mean_acc = cross_validation(splitted, knn, normalize=bool, classification=True)
            accuracies1, mean_acc1 = cross_validation(splitted, knn1, normalize=bool, classification=True)

            # print(mean_acc)
            print("Distance based: ", mean_acc, accuracies)
            print("Weighted: ", mean_acc1, accuracies1)

            knn2 = KNNRegression(k=k_val)
            knn3 = WeightedKNNRegression(k=k_val)

            mae_values, mean_mae = cross_validation(regression_splitted, knn2, normalize=bool, classification=False)
            mae_values1, mean_mae1 = cross_validation(regression_splitted, knn3, normalize=bool, classification=False)

            print("Mae values: ", mae_values)
            print("Mean mae: ", mean_mae)

            print("Weighted Mae values: ", mae_values1)
            print("Weighted Mean mae: ", mean_mae1)


def main():
    generalTest()

if __name__ == "__main__":
    main()
