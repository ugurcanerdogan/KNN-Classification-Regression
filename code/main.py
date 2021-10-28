import sys

import pandas as pd
from tqdm import tqdm

from Knn import KNN
from KnnRegression import KNNRegression
from WeightedKnn import WeightedKNN
from WeightedKnnRegression import WeightedKNNRegression
from utils import *


def generalTest():
    results_txt = open("./results.txt", "a")
    results_txt.write("----------------------------------------NEW RUN----------------------------------------\n\n")
    sys.stdout = results_txt

    # classification part
    classification_data = pd.read_csv("../glass.csv")
    classification_data = np.array(classification_data)

    # 5 fold cross validation
    splitted = k_fold_cross_validation_split(classification_data, 5)

    # regression part
    regression_data = pd.read_csv("../Concrete_Data_Yeh.csv")
    regression_data = np.array(regression_data)

    # 5 fold cross validation
    regression_splitted = k_fold_cross_validation_split(regression_data, 5)

    k_list_withWeight = []
    k_list_withoutWeight = []
    k_list_withWeight_nonNormalized = []
    k_list_withoutWeight_nonNormalized = []
    k_list_withWeight_MAE = []
    k_list_withoutWeight_MAE = []
    k_list_withWeight_nonNormalized_MAE = []
    k_list_withoutWeight_nonNormalized_MAE = []
    with tqdm(total=100) as pbar:
        for i in range(1, 6):
            k_val = (2 * i) - 1

            # updating tqdm progress
            pbar.set_description(f"In progress(k={k_val})")
            pbar.update(100 / 5)

            for bool in [True, False]:
                text = "With" if bool else "Without"
                print("*********************************")
                print(f"{text} Normalization")
                print(f"With k={k_val}")

                print("Classification Part")
                knn = KNN(k=k_val)
                knn1 = WeightedKNN(k=k_val)

                accuracies, mean_acc = cross_validation(splitted, knn, normalize=bool, classification=True)
                accuracies1, mean_acc1 = cross_validation(splitted, knn1, normalize=bool, classification=True)
                print("Distance based accuracies: ", accuracies, mean_acc)
                print("Weighted accuracies: ", accuracies1, mean_acc1)

                print("Regression Part")
                knn2 = KNNRegression(k=k_val)
                knn3 = WeightedKNNRegression(k=k_val)

                mae_values, mean_mae = cross_validation(regression_splitted, knn2, normalize=bool, classification=False)
                mae_values1, mean_mae1 = cross_validation(regression_splitted, knn3, normalize=bool,
                                                          classification=False)

                print("Mae values: ", mae_values)
                print("Weighted Mae values: ", mae_values1)

                if bool:
                    k_list_withoutWeight.append(mean_acc)
                    k_list_withWeight.append(mean_acc1)
                    k_list_withoutWeight_MAE.append(mean_mae)
                    k_list_withWeight_MAE.append(mean_mae1)
                else:
                    k_list_withoutWeight_nonNormalized.append(mean_acc)
                    k_list_withWeight_nonNormalized.append(mean_acc1)
                    k_list_withoutWeight_nonNormalized_MAE.append(mean_mae)
                    k_list_withWeight_nonNormalized_MAE.append(mean_mae1)

        plot_k_values_and_accuracies("KNN without weights", "KNN with weights", k_list_withoutWeight, k_list_withWeight,
                                     {'size': 12})
        plot_k_values_and_accuracies("KNN without weights (non normalized)", "KNN with weights (non normalized)",
                                     k_list_withoutWeight_nonNormalized, k_list_withWeight_nonNormalized, {'size': 9})
        plot_k_values_and_accuracies("KNN Regression without weights", "KNN Regression with weights",
                                     k_list_withoutWeight_MAE, k_list_withWeight_MAE, {'size': 10}, isAcc=False)
        plot_k_values_and_accuracies("KNN Regression without weights (non normalized)",
                                     "KNN Regression with weights (non normalized)",
                                     k_list_withoutWeight_nonNormalized_MAE, k_list_withoutWeight_nonNormalized_MAE,
                                     {'size': 7}, isAcc=False)

    results_txt.close()


def main():
    generalTest()


if __name__ == "__main__":
    main()
