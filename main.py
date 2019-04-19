from controllers.auth import spotify_login
from controllers.playlist import PlaylistController
from environment.variables import *
from models.playlist import PlaylistTrainer
from views.eda import ExploratoryDataAnalysis


if __name__ == '__main__':
    sp_client = spotify_login(username=SPOTIFY_USERNAME,
                              client_id=SPOTIFY_CLIENT_ID,
                              secret_id=SPOTIFY_SECRET_ID,
                              scope=SPOTIFY_SCOPE,
                              redirect_uri=SPOTIFY_REDIRECT_URI)

    pl = PlaylistController(client=sp_client)
    likedPlaylist = pl.get_playlist(user_id=SPOTIFY_USER_ID, playlist_id=LIKED_PLAYLIST_ID, liked=True)
    dislikedPlaylist = pl.get_playlist(user_id=SPOTIFY_USER_ID, playlist_id=DISLIKED_PLAYLIST_ID, liked=False)
    data = likedPlaylist.append(dislikedPlaylist, ignore_index=True)
    exploratory_view = ExploratoryDataAnalysis(data=data)
    exploratory_view.plot_all_views()
    input('Press enter on the console to exit...')
    trainer = PlaylistTrainer(data=data)


