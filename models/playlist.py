from pandas import DataFrame
from matplotlib import pyplot as plot
from sklearn.model_selection import train_test_split
from controllers.console import console


class PlaylistTrainer:

    @console('Splitting the dataset into training and test...')
    def __init__(self, data: DataFrame, size=0.25):
        self.data = data
        self.train, self.test = train_test_split(data, test_size=size)
