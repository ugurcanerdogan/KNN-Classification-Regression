from math import sqrt

import numpy as np
import numpy as np
import matplotlib.pyplot as plt

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

    # Comparison between predictions and the test labels.
    accuracy = (predicted == labels).sum() / len(predicted)
    # print(accuracy*100)
    return accuracy * 100


def mean_absolute_error(predicted, labels):
    predicted = np.array(predicted)
    labels = np.array(labels)

    mae = np.absolute(labels - predicted).sum() / len(predicted)
    return mae


def min_max_normalization(data):
    # max_bucket = [-1000000000 for i in range(len(data[0]))]

    # working on temp data
    temp_data = data.copy()

    length = len(temp_data[0])

    # scaling each attribute of the data
    for i in range(length):
        col_data = temp_data[:, i]
        _max = np.max(col_data)
        _min = np.min(col_data)

        # applying min-max normalization
        col_data = (col_data - _min) / (_max - _min)

        # replace data
        temp_data[:, i] = col_data

    return temp_data


def cross_validation(splitted_data, knn, normalize, classification):
    accuracies = []
    mae_values = []

    for data in splitted_data:
        sample_train = data[0]
        sample_test = data[1]

        # print(sample_train.shape)
        # print(sample_test.shape)

        # train and test sets
        X_train = sample_train[:, :-1]
        y_train = sample_train[:, -1]

        X_test = sample_test[:, :-1]
        y_test = sample_test[:, -1]

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

        # print(knn.y_train)
        # print(predictions)
        # print(y_test)

        if classification:
            # calculate accuracy
            accuracy = accuracy_score(predictions, y_test)
            accuracies.append(accuracy)
        else:
            mae = mean_absolute_error(predictions, y_test)
            mae_values.append(mae)

    if classification:
        accuracies = np.array(accuracies)
        return np.sort(accuracies),np.mean(accuracies)
    else:
        return np.sort(mae_values),np.mean(mae_values)


def plot_k_values(arr1):

    x = np.array([1, 3, 5, 7, 9])
    plt.plot(x,arr1,marker = 'o')

    plt.title("KNN")
    plt.xlabel("K values")
    plt.ylabel("Accuracies")

    plt.show()


def k_fold_cross_validation_split(X, k=5):
    # takes numpy array as an argument
    # shuffle
    np.random.shuffle(X)

    splitted_data = []
    data_partition_num = len(X) / k

    for i in range(k):
        start = int(i * data_partition_num)
        end = int((i + 1) * data_partition_num)

        test_set = X[start:end, :]
        train_set = np.concatenate((X[0:start, :], X[end:, :]), axis=0)

        splitted_data.append(np.array([train_set, test_set], dtype=object))

    return np.array(splitted_data)
