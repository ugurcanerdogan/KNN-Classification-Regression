from utils import *
from collections import defaultdict, Counter

import sys

class WeightedKNN:

    # setting k value of KNN algorithm
    def __init__(self, k=3):
        self.k = k

    # setting attribute and class data
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # majority voting function
    def vote(self, neighbours, frequence_array):
        distances = []
        classes = []
        indexes = []
        weights = []

        for neighbour in neighbours:
            _index = neighbour[1]    # neighbour[1] : index of related data's row
                                     # recall --> [dist, index_of_neighbour]

            _class = self.y_train[_index]    # _ class : corresponding data in the Class set
            _distance = neighbour[0]         # neighbour[0] : distance of related data
            _weight = 1 if _distance == 0 else (1/_distance)

            # save all information in various lists
            distances.append(_distance)
            classes.append(_class)
            indexes.append(_index)
            weights.append(_weight)

        # just for convention
        counted = frequence_array 


        # print("-----voting part-----")
        
        # old code
        #counted = list(Counter(classes).items())
        
        # print(classes)
        # print(counted)
        # print("distances: ", distances)

        most_common = counted[0]                    # find the most repetitive class
        class_numbers = [i[1] for i in counted]     # store the number of repetitions of classes
        # print(class_numbers)

        if class_numbers.count(most_common[0]) > 1: # if the most common class has more than one sample

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
                        # access distance of neighbour from the array which is created above
                        indiv_distance = distances[__index]

                        # store distances to calculate sums later and decide nearest neighbour
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

        
        
        # finding maximum class number to decide weight_dict size
        maximum_class = int(np.max([self.y_train[i[1]] for i in sorted_array[:self.k]]))
        
        weight_dict = {(i+1):0 for i in range(maximum_class+1)} 

        for _tuple in range(self.k):
            neighb_val = self.y_train[sorted_array[_tuple][1]]
            # print(sorted_array[_tuple][0])
            if not sorted_array[_tuple][0] == 0:
                weight_dict[neighb_val] += (1 / sorted_array[_tuple][0])
            else:
                print(sorted_array[_tuple][0])
                # if distance value equals to 0, weight value becomes 1 to avoid zero division error
                weight_dict[neighb_val] += 1
        
        # equivalent of Counter() method output from collections which has used on basic KNN
        """
        frequence_array = [
            (class1, weight1), 
            (class2, weight2),
            (..., ...),
            ...
        ]
        """
        frequence_array = [(_class, weight_dict[_class]) for _class in weight_dict.keys()]
        frequence_array = sorted(frequence_array, key=lambda x:x[1], reverse=True)

        # print(frequence_array)
        
        # print(sorted_array[:self.k])
        predicted = self.vote(sorted_array[:self.k], frequence_array)
        return predicted
