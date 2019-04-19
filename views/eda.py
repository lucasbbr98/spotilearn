# Exploratory Data Analysis (EDA)
from pandas import DataFrame
from matplotlib import pyplot as plot
import seaborn as sns
from controllers.console import console
from environment.variables import EDA_LIKED_COLOR_HEX, EDA_DISLIKED_COLOR_HEX

# plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
class ExploratoryDataAnalysis:
    def __init__(self, data: DataFrame):
        self.data = data
        self.liked = self.data[self.data['liked'] == 1]
        self.disliked = self.data[self.data['liked'] == 0]
        self.__color__()

    def plot_all_views(self):
        #self.plot_bpm()
        #self.plot_dance()
        self.plot_duration()
        self.plot_loudness()
        self.plot_speechiness()
        self.plot_valence()
        self.plot_energy()
        self.plot_acousticness()
        self.plot_key()
        self.plot_instrumentalness()


    @console('Plotting BPM Chart...')
    def plot_bpm(self):
        liked = self.liked['tempo']
        disliked = self.disliked['tempo']
        self.__plot__(liked=liked, disliked=disliked, name='BPM')

    @console('Plotting Danceability chart...')
    def plot_dance(self):
        liked = self.liked['danceability']
        disliked = self.disliked['danceability']
        self.__plot__(liked=liked, disliked=disliked, name='Danceability')

    @console('Plotting Duration(ms) chart...')
    def plot_duration(self):
        liked = self.liked['duration_ms']
        disliked = self.disliked['duration_ms']
        self.__plot__(liked=liked, disliked=disliked, name='Duration (ms)')

    @console('Plotting Loudness chart...')
    def plot_loudness(self):
        liked = self.liked['loudness']
        disliked = self.disliked['loudness']
        self.__plot__(liked=liked, disliked=disliked, name='Loudness')

    @console('Plotting Speechiness chart...')
    def plot_speechiness(self):
        liked = self.liked['speechiness']
        disliked = self.disliked['speechiness']
        self.__plot__(liked=liked, disliked=disliked, name='Speechiness')

    @console('Plotting Valence chart...')
    def plot_valence(self):
        liked = self.liked['valence']
        disliked = self.disliked['valence']
        self.__plot__(liked=liked, disliked=disliked, name='Valence')

    @console('Plotting Energy chart...')
    def plot_energy(self):
        liked = self.liked['energy']
        disliked = self.disliked['energy']
        self.__plot__(liked=liked, disliked=disliked, name='Energy')

    @console('Plotting Acousticness chart...')
    def plot_acousticness(self):
        liked = self.liked['acousticness']
        disliked = self.disliked['acousticness']
        self.__plot__(liked=liked, disliked=disliked, name='Acousticness')

    @console('Plotting Key chart...')
    def plot_key(self):
        liked = self.liked['key']
        disliked = self.disliked['key']
        self.__plot__(liked=liked, disliked=disliked, name='Key')

    @console('Plotting Instrumentalness chart...')
    def plot_instrumentalness(self):
        liked = self.liked['key']
        disliked = self.disliked['key']
        self.__plot__(liked=liked, disliked=disliked, name='Instrumentalness')

    @staticmethod
    def __color__():
        disliked = EDA_DISLIKED_COLOR_HEX  # Red
        liked = EDA_LIKED_COLOR_HEX  # Green
        colors = [disliked, liked]
        palette = sns.color_palette(colors)
        sns.set_palette(palette)
        sns.set_style('white')

    @staticmethod
    def __plot__(liked, disliked, name):
        plot.figure(figsize=(12, 6))
        plot.title("{0} Like/Dislike Distribution".format(name))
        plot.xlabel(name)
        plot.ylabel('Count')
        disliked.hist(alpha=0.7, bins=30)
        liked.hist(alpha=0.7, bins=30)
        plot.show(block=True)








