from pandas import DataFrame
from spotipy import Spotify
from controllers.console import console


class PlaylistController:
    def __init__(self, client: Spotify):
        self.client = client

    @console('Attempting to get playlist...')
    def get_playlist(self, user_id: str, playlist_id: str, liked: bool) -> DataFrame:
        if liked is None:
            raise ValueError("We must know if you like the playlist or not... ")

        songs = self._get_playlist_tracks(username=user_id, playlist_id=playlist_id)
        features, track_names = self._get_songs_features(songs)
        df = DataFrame(features, index=track_names)

        if liked:
            df['liked'] = 1
        else:
            df['liked'] = 0

        self._save_to_excel(df, liked)
        print('Playlist retrieved... items count: {0}'.format(len(df)))
        return df

    def _get_playlist_tracks(self, username, playlist_id):
        results = self.client.user_playlist_tracks(username, playlist_id)
        tracks = results['items']
        while results['next']:
            results = self.client.next(results)
            tracks.extend(results['items'])
        return tracks

    @console('Gettings all songs features... this might take a while :D')
    def _get_songs_features(self, songs):
        track_ids = []
        track_names = []

        for i in range(0, len(songs)):
            if songs[i]['track']['id']:  # Removes the local tracks in your playlist if there is any
                track_ids.append(songs[i]['track']['id'])
                track_names.append(songs[i]['track']['name'])

        features = []
        for i in range(0, len(track_ids)):
            audio_features = self.client.audio_features(track_ids[i])
            for track in audio_features:
                features.append(track)

        return features, track_names

    @console('Saving data in folder /database')
    def _save_to_excel(self, df, liked):
        try:
            if liked:
                df.to_excel("database/liked.xlsx")
            else:
                df.to_excel("database/disliked.xlsx")
        except Exception as e:
            print('Skipping save... Failed due to exception:')
            print(e)

