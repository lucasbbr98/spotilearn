import pandas as pd
import numpy as np
from models.ml.first_neural import FeedForwardSigmoidNeuralNetwork
from api.static.preprocessing import standard_features, train_test_split
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read Spotify Dataset
df = pd.read_excel('output_shuffled.xlsx').drop_duplicates()
x_data = df.iloc[:, 0:9].to_numpy()
y_data = df.iloc[:, 13].to_numpy()

# Pre processing
x_data = standard_features(x=x_data)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
n_inputs = x_train.shape[1]
ffn = FeedForwardSigmoidNeuralNetwork(n_inputs=9, hidden_sizes=[9, 2])

ffn.fit(x_train, y_train, epochs=500, learning_rate=.01, display_loss=True)
y_pred_train = ffn.predict(x_train)
y_pred_binarised_train = (y_pred_train >= 0.5).astype("int").ravel()
y_pred_val = ffn.predict(x_test)
y_pred_binarised_val = (y_pred_val >= 0.5).astype("int").ravel()
accuracy_train = accuracy_score(y_pred_binarised_train, y_train)
accuracy_val = accuracy_score(y_pred_binarised_val, y_test)

ffn.plot_roc(y_test= y_test, y_pred=y_pred_binarised_val)
#model performance
print("Training accuracy", round(accuracy_train, 2))
print("Validation accuracy", round(accuracy_val, 2))

