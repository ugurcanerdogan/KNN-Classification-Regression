from utils import *
from collections import defaultdict, Counter


class WeightedKNN:

    def __init__(self, k=3):
        # partition sayısı seç
        self.k = k

    def fit(self, X, y):
        # trainin datasını ve class numunu seç
        self.X_train = X
        self.y_train = y

    def vote(self, neighbours, weight_dict):
        distances = []
        classes = []
        indexes = []

        for neighbour in neighbours:
            index = neighbour[1]

            _class = self.y_train[index]
            _distance = neighbour[0]

            distances.append(_distance)
            classes.append(_class)
            indexes.append(index)


        # print("-----voting part-----")
        counted = list(Counter(classes).items())
        # print(classes)
        # print(counted)
        # print("distances: ", distances)

        most_common = counted[0]
        class_numbers = [i[1] for i in counted]
        # print(class_numbers)

        predicted_class = None
        if class_numbers.count(most_common[0]) > 1:
            # x = input("tie found, press any key to continue")
            # that means we have a tie situation
            classes_of_duplicate_occurrences = []
            for cnt in counted:
                if cnt[1] == most_common[1]:
                    # the one that we are looking for
                    classes_of_duplicate_occurrences.append(cnt[0])

            # print(classes_of_duplicate_occurrences)

            # breaking tie part
            new_distances = [[] for i in range(int(max(classes_of_duplicate_occurrences)) + 1)]

            for _class in classes_of_duplicate_occurrences:
                for __index, __class in enumerate(classes):
                    if _class == __class:
                        # accessing distance of neighbour from the array which is created above
                        indiv_distance = distances[__index]

                        # storing distances to calculate sums later and decide nearest neighbour
                        new_distances[int(_class)].append(indiv_distance)

            # print(new_distances)

            empty_list_indices = [i for i in range(len(new_distances)) if len(new_distances[i]) == 0]

            for index, new_distance in enumerate(new_distances):
                new_distances[index] = sum(new_distance)

            new_distances = np.array(new_distances)
            # print(new_distances)


            sorted_indices = np.argsort(new_distances)
            # print(sorted_indices)

            # print("empty part")
            # print(empty_list_indices)

            will_be_deleted_indices = []
            for index in empty_list_indices:
                for i, _index in enumerate(sorted_indices):
                    if index == _index:
                        will_be_deleted_indices.append(i)

            sorted_indices = np.delete(sorted_indices, will_be_deleted_indices)
            # print(sorted_indices)

            predicted_class = int(sorted_indices[0])
            # print(predicted_class)
        else:
            # no tie select most common as nearest neighbour
            predicted_class = int(most_common[0])

        return predicted_class

    def predict(self, X):
        predicted_classes = [self.predict_helper(row) for row in X]
        return predicted_classes

    def predict_helper(self, x):
        """
        [
            [dist, index_of_neighbour],
            [another_dist, index_of_another_neighbour],
            ...
        ]
        """

        # distance calculation
        distances = [[euclidean_distance(x, self.X_train[i]), i] for i in range(len(self.X_train))]

        # sorting to find nearest neighbour
        # print(distances)
        sorted_array = sorted(distances, key=lambda x: x[0])
        # print(sorted_array)

        # print("Asked data: ", x.astype(int))
        # print("Nearest Neighbours:")
        for m in range(self.k):
            index = sorted_array[m][1]
            # print(self.X_train[index], ", class: " ,self.y_train[index])

        ## ADDED WEIGHT DICTIONARY HERE

        weight_dict = {1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0, 6.0: 0, 7.0: 0}
        for _tuple in range(self.k):
            neighb_val = self.y_train[sorted_array[_tuple][1]]
            if not sorted_array[_tuple][0] == 0:
                weight_dict[neighb_val] += (1 / sorted_array[_tuple][0])

        predicted = self.vote(sorted_array[:self.k], weight_dict)
        return predicted