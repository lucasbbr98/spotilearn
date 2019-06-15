import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from models.ml.feed_forward_neural_network import FeedForwardSigmoidNeuralNetwork
from api.static.preprocessing import standard_features, train_test_split
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Read Spotify Dataset
df = pd.read_excel('output_shuffled.xlsx').drop_duplicates()
x_data = df.iloc[:, 0:9].to_numpy()
y_data = df.iloc[:, 13].to_numpy()

# Pre processing
x_data = standard_features(x=x_data)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

forest = RandomForestClassifier(n_estimators=500, max_depth= 5)
forest.fit(x_train, y_train)
score = forest.score(x_test, y_test)

print(score)