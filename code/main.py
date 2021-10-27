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
    k_list = list()

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

                accuracies,mean_acc = cross_validation(splitted, knn, normalize=bool, classification=True)
                accuracies1,mean_acc1 = cross_validation(splitted, knn1, normalize=bool, classification=True)
                print("Distance based accuracies: ", accuracies,mean_acc)
                print("Weighted accuracies: ", accuracies1, mean_acc1)

                if bool:
                    k_list.append(mean_acc)



                # print("Regression Part")
                # knn2 = KNNRegression(k=k_val)
                # knn3 = WeightedKNNRegression(k=k_val)
                #
                # mae_values,mean_mae = cross_validation(regression_splitted, knn2, normalize=bool, classification=False)
                # mae_values1,mean_mae1 = cross_validation(regression_splitted, knn3, normalize=bool, classification=False)
                #
                # print("Mae values: ", mae_values)
                # print("Weighted Mae values: ", mae_values1)

        plot_k_values(k_list)
    results_txt.close()


def main():
    generalTest()


if __name__ == "__main__":
    main()
