import numpy as np
import matplotlib.pyplot as plt
from api.static.visualization import draw_neural_net
from models.ml.visualization.neural_network import NeuralNetworkDrawer
import sklearn.metrics as metrics

np.random.seed(100)

'''
HOW TO USE:
Python
nn = NeuralNetwork()
nn.add_layer(Layer(2, 3, 'tanh'))
nn.add_layer(Layer(3, 3, 'sigmoid'))
nn.add_layer(Layer(3, 2, 'sigmoid'))

# Define dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

# Train the neural network
errors = nn.train(X, y, 0.3, 290, plot_errors = True)
print('Accuracy: %.2f%%' % (nn.accuracy(nn.predict(X), y.flatten()) * 100))

'''
class Layer:
    """
    Represents a layer (hidden or output) in our neural network.
    """

    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None, name=None):
        """
        :param int n_input: The input size (coming from the input layer or a previous hidden layer)
        :param int n_neurons: The number of neurons in this layer.
        :param str activation: The activation function to use (if any).
        :param weights: The layer's weights.
        :param bias: The layer's bias.
        """
        self.inputs = n_input
        self.neurons = n_neurons
        self.name = name
        self.weights = weights if weights is not None else np.random.rand(n_input, n_neurons)
        self.activation = activation.lower().strip()
        self.bias = bias if bias is not None else np.ones(n_neurons)
        self.last_activation = None
        self.error = None
        self.delta = None
        self.epoch = 0

    def activate(self, x):
        """
        Calculates the dot product of this layer.
        :param x: The input.
        :return: The result.
        """

        r = np.dot(x, self.weights) + self.bias
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    def _apply_activation(self, r):
        """
        Applies the chosen activation function (if any).
        :param r: The normal value.
        :return: The "activated" value.
        """

        # In case no activation function was chosen
        if self.activation is None:
            return r

        elif self.activation == 'tanh':
            return np.tanh(r)

        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))

        elif self.activation == 'relu':
            return np.maximum(0.0, r)

        else:
            raise NotImplementedError("Activation function not implemented yet...")

        return r

    def apply_activation_derivative(self, r):
        """
        Applies the derivative of the activation function (if any).
        :param r: The normal value.
        :return: The "derived" value.
        """

        # We use 'r' directly here because its already activated, the only values that
        # are used in this function are the last activations that were saved.

        if self.activation is None:
            return r
        elif self.activation == 'tanh':
            return 1 - r ** 2

        elif self.activation == 'sigmoid':
            return r * (1 - r)

        elif self.activation == 'relu':
            return 1. * (r > 0)
        else:
            raise NotImplementedError("Activation function not implemented yet...")

        return r


class NeuralNetwork:
    """
    Represents a neural network.
    """

    def __init__(self):
        self._layers = []
        self._errors = []
        self.epoch = 0

    def add_layer(self, layer):
        """
        Adds a layer to the neural network.
        :param Layer layer: The layer to add.
        """

        self._layers.append(layer)

    def feed_forward(self, X):
        """
        Feed forward the input through the layers.
        :param X: The input values.
        :return: The result.
        """

        for layer in self._layers:
            X = layer.activate(X)

        return X

    def predict(self, X):
        """
        Predicts a class (or classes).
        :param X: The input values.
        :return: The predictions.
        """

        ff = self.feed_forward(X)

        # One row
        if ff.ndim == 1:
            return np.argmax(ff)

        # Multiple rows
        return np.argmax(ff, axis=1)

    def backpropagation(self, X, y, learning_rate):
        """
        Performs the backward propagation algorithm and updates the layers weights.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        """

        # Feed forward for the output
        output = self.feed_forward(X)

        # Loop over the layers backward
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]

            # If this is the output layer
            if layer == self._layers[-1]:
                layer.error = y - output
                # The output = layer.last_activation in this case
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                k = layer.apply_activation_derivative(layer.last_activation)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)

        # Update the weights
        for i in range(len(self._layers)):
            layer = self._layers[i]
            # The input is either the previous layers output or X itself (for the first hidden layer)
            input_to_use = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            layer.weights += layer.delta * input_to_use.T * learning_rate

    def train(self, X, y, x_test, y_test, learning_rate, max_epochs, plot_errors=False, plot_accuracies=False):
        """
        Trains the neural network using backpropagation.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        :param int max_epochs: The maximum number of epochs (cycles).
        :return: The list of calculated MSE errors.
        """
        self.epoch = max_epochs
        mses = []
        test_mses = []
        train_accuracies = []
        test_accuracies = []
        for i in range(max_epochs):
            for j in range(len(X)):
                self.backpropagation(X[j], y[j], learning_rate)

            mse = np.mean(np.square(y - self.feed_forward(X)))
            mses.append(mse)
            test_mse = np.mean(np.square(y_test - self.feed_forward(x_test)))
            test_mses.append(test_mse)
            train_accuracies.append(self.accuracy(X, y))
            test_accuracies.append(self.accuracy(x_test, y_test))

        self._errors = mses
        if plot_errors:
            plt.plot(mses, 'r', label='train')
            plt.plot(test_mses, 'g', label='test')
            plt.title('Train and Test Cost Function')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Square Error')
            plt.legend(loc='upper left')
            plt.show()

        if plot_accuracies:
            plt.plot(train_accuracies, 'r', label='train')
            plt.plot(test_accuracies, 'g', label='test')
            plt.title('Train and Test Accuracies by Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(loc='upper left')
            plt.show()

        return mses

    def accuracy(self, x, y):
        """
        Calculates the accuracy between the predicted labels and true labels.
        :param y_pred: The predicted labels.
        :param y_true: The true labels.
        :return: The calculated accuracy.
        """
        y_pred = self.predict(x)
        y_true = y.flatten()
        return (y_pred == y_true).mean()


    def visualize(self):
        layer_sizes = []
        layer_weights = []
        for index, layer in enumerate(self._layers):
            if index == 0:
                layer_sizes.append(layer.inputs)
            layer_sizes.append(layer.neurons)
            layer_weights.append(w for w in layer.weights)


        #network = NeuralNetworkDrawer(layer_sizes)
        #network.draw()

        # Need empty strings for unlabeled nodes at start, but not at end
        '''node_text = ['bias', 'acoustic', 'dance', 'duration', 'energy', 'instruments', 'key', 'liveness', 'loudness',
                     'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24',
                     'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20',
                     'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12',
                     '1', '0']'''
        node_text = ['bias', 'acoustic', 'dance', 'duration', 'energy', 'instruments', 'key', 'liveness', 'loudness',
                     'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15',
                     'a16', 'a17', 'a18',
                     'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9',
                     '1', '0']

        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca()
        draw_neural_net(ax, .1, .9, .9, .1, layer_sizes, node_text)
        plt.show()


    def roc_curve(self, x_test, y_test):
        # calculate the fpr and tpr for all thresholds of the classification
        y_predicted = self.predict(x_test)
        preds = y_predicted
        fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        roc_auc = metrics.auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()