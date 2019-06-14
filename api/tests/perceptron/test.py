import pandas as pd
import numpy as np
from models.ml.perceptron import Perceptron
from sklearn.linear_model import Perceptron as SKPerceptron
from api.static.preprocessing import add_ones, standard_features, train_test_split
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


if __name__ == '__main__':

    # Read Spotify Dataset
    df = pd.read_excel('output_shuffled.xlsx').drop_duplicates()
    x_data = df.iloc[:, 0:9].to_numpy()
    y_data = df.iloc[:, 13].to_numpy()


    # Pre processing
    x_data = standard_features(x=x_data)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

    # API
    perceptron = Perceptron(learning_rate=0.001, max_iterations=50)
    perceptron.fit(train_x=x_train, train_y=y_train)
    predictions = perceptron.predict(test=x_test)
    metrics = perceptron.metrics.calculate(y_test)
    perceptron.graphs.plot_loss_epoch()
    perceptron.graphs.plot_roc()

    print('API\nAccuracy: {0}%\nROC_AUC: {1}\nPrecision: {2}\nRecall: {3}\nF1 Score: {4}\n\n'
          .format(metrics.accuracy * 100, metrics.roc_auc, metrics.precision, metrics.recall, metrics.f1))


    # SK LEARN
    perceptron = SKPerceptron(eta0=0.001, max_iter=500, tol=0.01)
    perceptron.fit(x_train, y_train)
    predictions = perceptron.predict(x_test)
    score = perceptron.score(x_test, y_test)
    print('SK Learn Score: {0}%'.format(score * 100))




