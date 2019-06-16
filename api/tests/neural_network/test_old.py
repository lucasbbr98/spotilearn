import pandas as pd
import numpy as np
from models.ml.feed_forward_neural_network import FeedForwardSigmoidNeuralNetwork
from models.ml.backwards_propagation_neural_network import NeuralNetwork, Layer
from api.static.preprocessing import standard_features, train_test_split
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numba import njit

# Read Spotify Dataset
df = pd.read_excel('output_shuffled.xlsx').drop_duplicates()
x_data = df.iloc[:, 0:9].to_numpy()
y_data = df.iloc[:, 13].to_numpy()

# Pre processing
x_data = standard_features(x=x_data)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
n_inputs = x_train.shape[1]
ffn = FeedForwardSigmoidNeuralNetwork(n_inputs=9, hidden_sizes=[12])
ffn.fit(x_train, y_train, x_test, y_test, epochs=600, learning_rate=.01, display_loss=True, display_accuracy=True)
y_pred_train = ffn.predict(x_train)
y_pred_binarised_train = (y_pred_train >= 0.5).astype("int").ravel()
y_pred_val = ffn.predict(x_test)
y_pred_binarised_val = (y_pred_val >= 0.5).astype("int").ravel()
accuracy_train = accuracy_score(y_pred_binarised_train, y_train)
accuracy_val = accuracy_score(y_pred_binarised_val, y_test)

ffn.plot_roc(y_test= y_test, y_pred=y_pred_binarised_val)
#model performance
print("Training accuracy", round(accuracy_train, 4))
print("Validation accuracy", round(accuracy_val, 4))


def test_hidden_layers_1():
    size1 = 1
    results = []
    for i in range(100):
        ffn = FeedForwardSigmoidNeuralNetwork(n_inputs=9, hidden_sizes=[22])
        ffn.fit(x_train, y_train, epochs=100, learning_rate=.01, display_loss=False)
        y_pred_train = ffn.predict(x_train)
        y_pred_binarised_train = (y_pred_train >= 0.5).astype("int").ravel()
        y_pred_val = ffn.predict(x_test)
        y_pred_binarised_val = (y_pred_val >= 0.5).astype("int").ravel()
        accuracy_train = accuracy_score(y_pred_binarised_train, y_train)
        accuracy_val = accuracy_score(y_pred_binarised_val, y_test)
        results.append((size1, accuracy_train, accuracy_val, min(ffn.loss)))

        size1 = size1 + 1
        print(i)

    pd.DataFrame(list(results), columns=['size', 'Accuracy Train', 'Accuracy Test', 'Minimum Loss']).to_excel('results-a_99-h-1.xlsx')

def test_epoch_layers_1():
    epochs = 1
    results = []
    for i in range(101):
        ffn = FeedForwardSigmoidNeuralNetwork(n_inputs=9, hidden_sizes=[100])
        ffn.fit(x_train, y_train, epochs=epochs, learning_rate=.01, display_loss=False)
        y_pred_train = ffn.predict(x_train)
        y_pred_binarised_train = (y_pred_train >= 0.5).astype("int").ravel()
        y_pred_val = ffn.predict(x_test)
        y_pred_binarised_val = (y_pred_val >= 0.5).astype("int").ravel()
        accuracy_train = accuracy_score(y_pred_binarised_train, y_train)
        accuracy_val = accuracy_score(y_pred_binarised_val, y_test)
        results.append((epochs, accuracy_train, accuracy_val, min(ffn.loss)))

        if epochs == 1:
            epochs = epochs + 9
        else:
            epochs = epochs + 10

        print(i)

    pd.DataFrame(list(results), columns=['Epochs', 'Accuracy Train', 'Accuracy Test', 'Minimum Loss']).to_excel('results-e-h-1-100.xlsx')


#test_epoch_layers_1()
