#Author: Peter Gonzalez
import numpy as np

class BoostingAlgorithm():

    def __init__(self, n_rounds=4):
        self.n_rounds = n_rounds

    #Ada Boosting
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.full(n_samples, (1 / n_samples))
        self.classifiers = []
        for index in range(self.n_rounds):
            clf = classifiers()
            minimum_error = float('inf')
            for feature in range(n_features):
                X_column = X[:, feature]
                thresholds = np.unique(X_column)
                for line in thresholds:
                    classifier_label = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column == 0] = -1
                    misclassified = self.weights[y != predictions]
                    error = sum(misclassified)
                    if error > 0.5:
                        classifier_label = -1
                        error = 1 - error
                    if error < minimum_error:
                        minimum_error = error
                        clf.classifier_label = classifier_label
                        clf.line = line
                        clf.top_idx = feature
            clf.alpha = 0.5 * np.log((1.0 - minimum_error + 1e-10) / (minimum_error + 1e-10))
            predictions = clf.predict(X)
            self.weights *= np.exp(-clf.alpha * y * predictions)
            self.weights /= np.sum(self.weights)
            self.classifiers.append(clf)

    def predict(self, X):
        classifier_predictions = [clf.alpha * clf.predict(X) for clf in self.classifiers]
        y_predictions = np.sum(classifier_predictions, axis=0)
        y_predictions = np.sign(y_predictions)
        return y_predictions



class classifiers():
    def __init__(self):
        self.line = None
        self.alpha = None
        self.classifier_label = 1
        self.top_idx = None
        

    def predict(self, X):
        n_samples, n_features = X.shape
        X_column = X[:, self.top_idx]
        predictions = np.ones(n_samples)
        if self.classifier_label == 1:
            predictions[X_column == 0] = -1
        else:
            predictions[X_column == 1] = -1
        return predictions