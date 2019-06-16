import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import  mean_squared_error
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score

#LEGAL 1
#https://blog.zhaytam.com/2018/08/15/implement-neural-network-backpropagation/
#https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/


class FeedForwardSigmoidNeuralNetwork:
    def __init__(self, n_inputs, hidden_sizes=[2, 3]):
        # intialize the inputs
        self.nx = n_inputs
        self.ny = 1
        self.nh = len(hidden_sizes)
        self.sizes = [self.nx] + hidden_sizes + [self.ny]

        self.W = {}
        self.B = {}
        self.loss = []
        for i in range(self.nh + 1):
            self.W[i + 1] = np.random.randn(self.sizes[i], self.sizes[i + 1])
            self.B[i + 1] = np.ones((1, self.sizes[i + 1]))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


    def forward_pass(self, x):
        self.A = {}
        self.H = {}
        self.H[0] = x.reshape(1, -1)
        for i in range(self.nh + 1):
            self.A[i + 1] = np.matmul(self.H[i], self.W[i + 1]) + self.B[i + 1]
            self.H[i + 1] = self.sigmoid(self.A[i + 1])
        return self.H[self.nh + 1]

    def grad_sigmoid(self, x):
        return x * (1 - x)


    def grad(self, x, y):
        self.forward_pass(x)
        self.dW = {}
        self.dB = {}
        self.dH = {}
        self.dA = {}
        L = self.nh + 1
        self.dA[L] = (self.H[L] - y)
        for k in range(L, 0, -1):
            self.dW[k] = np.matmul(self.H[k - 1].T, self.dA[k])
            self.dB[k] = self.dA[k]
            self.dH[k - 1] = np.matmul(self.dA[k], self.W[k].T)
            self.dA[k - 1] = np.multiply(self.dH[k - 1], self.grad_sigmoid(self.H[k - 1]))


    def fit(self, X, Y, x_test, y_test, epochs=1, learning_rate=1, initialise=True, display_loss=False, display_accuracy=False):

        # initialise w, b
        if initialise:
            for i in range(self.nh + 1):
                self.W[i + 1] = np.random.randn(self.sizes[i], self.sizes[i + 1])
                self.B[i + 1] = np.zeros((1, self.sizes[i + 1]))

        loss = {}
        accuracy_trains=[]
        accuracy_tests=[]
        for e in range(epochs):
            dW = {}
            dB = {}
            for i in range(self.nh + 1):
                dW[i + 1] = np.zeros((self.sizes[i], self.sizes[i + 1]))
                dB[i + 1] = np.zeros((1, self.sizes[i + 1]))
            for x, y in zip(X, Y):
                self.grad(x, y)
                for i in range(self.nh + 1):
                    dW[i + 1] += self.dW[i + 1]
                    dB[i + 1] += self.dB[i + 1]

            m = X.shape[1]
            for i in range(self.nh + 1):
                self.W[i + 1] -= learning_rate * dW[i + 1] / m
                self.B[i + 1] -= learning_rate * dB[i + 1] / m

            Y_pred = self.predict(X)
            loss[e] = mean_squared_error(Y_pred, Y)
            y_pred_train = self.predict(X)
            y_pred_binarised_train = (y_pred_train >= 0.5).astype("int").ravel()
            y_pred_val = self.predict(x_test)
            y_pred_binarised_val = (y_pred_val >= 0.5).astype("int").ravel()
            accuracy_trains.append(accuracy_score(y_pred_binarised_train, Y))
            accuracy_tests.append(accuracy_score(y_pred_binarised_val, y_test))

        self.loss = list(loss.values())
        if display_loss:
            plt.plot(list(loss.values()))
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error')
            plt.show()

        if display_accuracy:
            plt.plot(accuracy_trains, 'r', label='train')
            plt.plot(accuracy_tests, 'g', label='test')
            plt.title('Train and Test Accuracies by Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(loc='upper left')
            plt.show()


    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred).squeeze()

    def plot_roc(self, y_test, y_pred, show=True):
        plt.subplots(figsize=(10, 6))
        plt.title('Receiver Operating Characteristic')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='diagonal')
        fpr, tpr, thresh = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.plot(fpr, tpr, 'b', label="ROC Curve, auc=" + str(roc_auc))
        if show:
            plt.show()