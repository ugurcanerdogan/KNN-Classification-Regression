class KnnBase:

    # setting k value of KNN algorithm
    def __init__(self, k=3):
        self.k = k

    # setting attribute and class data
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # majority voting function
    def vote(self, neighbours, frequence_array):
        raise NotImplementedError("Each class must implement their own voting function !")

    def predict(self, X):
        predicted_classes = [self.predict_helper(row) for row in X]
        return predicted_classes

    def predict_helper(self, x):
        raise NotImplementedError("Each class must implement their own predict_helper function !")
