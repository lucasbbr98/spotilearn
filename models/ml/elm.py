import numpy as np
import pandas as pd


class ELMRegressor:
    def __init__(self, n_hidden_units):
        self.n_hidden_units = n_hidden_units
        self.random_weights = None
        self.w_elm = None

    def fit(self, X, y):
        X = np.column_stack([X, np.ones([X.shape[0], 1])])
        self.random_weights = np.random.randn(X.shape[1], self.n_hidden_units)
        G = np.tanh(X.dot(self.random_weights))
        self.w_elm = np.linalg.pinv(G).dot(y)
        return self.w_elm

    def predict(self, X):
        X = np.column_stack([X, np.ones([X.shape[0], 1])])
        G = np.tanh(X.dot(self.random_weights))
        return G.dot(self.w_elm)


if __name__ == '__main__':
    df = pd.read_excel('multiple_regression.xlsx')
    test_x = [24, 70]
    train_x = df.iloc[:, 0]
    train_y = df.iloc[:, 1:]
    elm = ELMRegressor(n_hidden_units=100)
    w = elm.fit(train_x, train_y)
    predicction = elm.predict(test_x)
