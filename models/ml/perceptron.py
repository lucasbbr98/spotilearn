import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from api.static.metrics import confusion_matrix


class Perceptron():
    def __init__(self, learning_rate=0.01, max_iterations=500, tolerance=0.001, threshold=0.0):
        self.l_rate = learning_rate
        self.epoch = max_iterations
        self.tolerance = tolerance
        self.threshold = threshold
        self.loss_history = []
        self.weights = None
        self.metrics = PerceptronMetrics(perceptron=self)
        self.graphs = PerceptronGraphs(perceptron=self, metrics=self.metrics)


    # Make a prediction with weights on a single row
    def _predict_row(self, row, weights):
        activation = weights[0]
        for i in range(len(row) - 1):
            activation += weights[i + 1] * row[i]
        return 1.0 if activation >= self.threshold else -1.0

    @staticmethod
    def check_accuracy(predictions, test_answers):
        corrects = 0
        for index, prediction in enumerate(predictions):
            if prediction == test_answers[index]:
                corrects = corrects + 1

        return corrects / float(len(test_answers)) * 100.0

    # Estimate Perceptron weights using stochastic gradient descent
    def train_weights(self, x, y):
        weights = [0.0 for i in range(len(x[0]))]
        for epoch in range(self.epoch):
            current_error = 0
            for index, row in enumerate(x):
                prediction = self._predict_row(row, weights)
                update = self.l_rate*(y[index] - prediction)
                current_error += int(update != 0.0)
                weights[0] = weights[0] + self.l_rate * update
                for i in range(len(row) - 1):
                    weights[i + 1] = weights[i + 1] + update * row[i]
            self.loss_history.append(current_error)
        return weights

    # Perceptron Algorithm With Stochastic Gradient Descent
    def fit(self, train_x, train_y):
        self.weights = self.train_weights(train_x, train_y)


    def predict(self, test):
        predictions = list()
        for row in test:
            prediction = self._predict_row(row, self.weights)
            predictions.append(prediction)

        self.metrics.predictions = predictions
        return predictions


class PerceptronMetrics:
    def __init__(self, perceptron):
        self.perceptron = perceptron
        self.y_test = None
        self.predictions = None

        # Metrics
        self.roc_auc = 0
        self.accuracy = 0
        self.recall = 0
        self.precision = 0
        self.f1 = 0
        self.confusion_matrix = None

    def calculate(self, y_test):
        self.y_test = y_test
        tp, tn, fp, fn = confusion_matrix(y_test, self.predictions)
        fpr, tpr, thresh = roc_curve(self.y_test, self.predictions)

        self.roc_auc = auc(fpr, tpr)
        self.precision = self.__calculate_precision__(tp, tn)
        self.recall = self.__calculate_recall__(tp, fn)
        self.accuracy = self.__calculate_accuracy__(tp, tn, fp, fn)
        self.f1 = self.__calculate_f1__(self.precision, self.recall)
        self.confusion_matrix = np.array([[tp, fp], [fn, tn]])
        return self

    @staticmethod
    def __calculate_precision__(tp, tn):
        try:
            return tp/(tp+tn)
        except:
            return 0

    @staticmethod
    def __calculate_recall__(tp, fn):
        try:
            return tp/(tp+fn)
        except:
            return 0

    @staticmethod
    def __calculate_accuracy__(tp, tn, fp, fn):
        try:
            return (tp+tn)/(fp+fn+tp+tn)
        except:
            return 0

    @staticmethod
    def __calculate_f1__(precision, recall):
        try:
            return 2 * precision * recall / (precision + recall)
        except:
            return 0


class PerceptronGraphs:
    def __init__(self, perceptron, metrics):
        self.perceptron = perceptron
        self.metrics = metrics

    def plot_loss_epoch(self, show=True):
        fig = plt.figure()
        plt.plot(np.arange(0, self.perceptron.epoch), self.perceptron.loss_history)
        fig.suptitle("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        if show:
            plt.show()

    def plot_roc(self, show=True):
        plt.subplots(figsize=(10, 6))
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='diagonal')
        fpr, tpr, thresh = roc_curve(self.metrics.y_test, self.metrics.predictions)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label="ROC Curve, auc=" + str(roc_auc))
        if show:
            plt.show()
