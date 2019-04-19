from spotipy import Spotify
import spotipy.util as util
from environment.variables import *
from controllers.console import console


@console('Logging in...')
def spotify_login(username, client_id, secret_id, redirect_uri, scope) -> Spotify:
    token = util.prompt_for_user_token(username, scope, client_id, secret_id, redirect_uri)
    if token:
        sp = Spotify(auth=token)
        usr = sp.current_user()
        print('Hi there {0}'.format(usr['display_name']))
        return sp
    else:
        raise ValueError("Can't get token for {0}".format(SPOTIFY_USERNAME))
