from flask import Blueprint, request, redirect, g, render_template, url_for, session
from flask_login import login_required, current_user, logout_user, login_user
from flask_socketio import SocketIO
from plugins.login import login_manager
from plugins.socket import socket_manager as socket
from time import sleep
from models.user import User
from models.playlist import Playlist
from constants.spotify import *
from urllib.parse import quote
import requests
import json
from datetime import datetime

web = Blueprint('web', __name__)
TASKS_MSGS = ''

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

@web.route('/')
def index():
    return render_template('index.html')


@web.route("/login")
def login():
    # Auth Step 1: Authorization
    url_args = "&".join(["{}={}".format(key, quote(val)) for key, val in auth_query_parameters.items()])
    auth_url = "{}/?{}".format(SPOTIFY_AUTH_URL, url_args)
    return redirect(auth_url)


@web.route("/callback/q")
def callback():
    # Auth Step 4: Requests refresh and access tokens
    auth_token = request.args['code']
    code_payload = {
        "grant_type": "authorization_code",
        "code": str(auth_token),
        "redirect_uri": REDIRECT_URI,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
    }
    post_request = requests.post(SPOTIFY_TOKEN_URL, data=code_payload)

    # Auth Step 5: Tokens are Returned to Application
    response_data = json.loads(post_request.text)
    access_token = response_data["access_token"]
    refresh_token = response_data["refresh_token"]
    token_type = response_data["token_type"]
    expires_in = response_data["expires_in"]

    # Auth Step 6: Use the access token to access Spotify API
    authorization_header = {"Authorization": "Bearer {}".format(access_token)}

    # Get profile data
    user_profile_api_endpoint = "{}/me".format(SPOTIFY_API_URL)
    profile_response = requests.get(user_profile_api_endpoint, headers=authorization_header)
    profile_data = json.loads(profile_response.text)
    user = User(spotify_id=profile_data['id'], name=profile_data['display_name'], jwt=access_token)
    user_check = User.query.filter_by(spotify_id=profile_data['id']).first()

    # Updates DB
    if not user_check:
        user.save()
    else:
        user_check.name = user.name
        user_check.spotify_id = user.spotify_id
        user_check.jwt = user.jwt
        user_check.save()
        user = user_check

    login_user(user)

    # Get user playlist data
    playlist_api_endpoint = "{}/playlists".format(profile_data["href"])
    playlists_response = requests.get(playlist_api_endpoint, headers=authorization_header)
    playlist_data = json.loads(playlists_response.text)

    playlists = []
    for item in playlist_data["items"]:
        playlists.append(Playlist(spotify_id=item['id'], name=item['name'], image=item['images'][0]['url']).to_json())

    session['playlists'] = playlists
    return redirect(url_for('web.like'))


@web.route('/logout')
def logout():
    logout_user()
    return render_template("index.html")


@web.route('/like')
@login_required
def like():
    liked_sid = request.args.get('sid')
    if liked_sid and liked_sid != '':
        session['liked_sid'] = liked_sid
        return redirect(url_for('web.dislike'))

    return render_template('like.html',  playlists=session['playlists'])


@web.route('/dislike')
@login_required
def dislike():
    playlists = session['playlists']
    liked_sid = session['liked_sid']
    if not liked_sid or liked_sid == '':
        return redirect(url_for('web.like'))

    for playlist in playlists:
        if playlist['spotify_id'] == liked_sid:
            playlists.remove(playlist)
            break

    disliked_sid = request.args.get('sid')
    if disliked_sid and disliked_sid != '':
        session['disliked_sid'] = disliked_sid
        return redirect(url_for('web.loading'))

    return render_template('dislike.html', playlists=session['playlists'])


@web.route('/loading')
@login_required
def loading():
    return render_template('loading.html')


@web.route('/fake_loading')
@login_required
def loading_status():
    sleep(10)
    TASKS_MSGS = {'msg': 'T1', 'time': str(datetime.today().strftime('%H:%M'))}
    sleep(10)
    TASKS_MSGS = {'msg': 'T2', 'time': str(datetime.today().strftime('%H:%M'))}
    sleep(10)
    TASKS_MSGS = {'msg': 'T3', 'time': str(datetime.today().strftime('%H:%M'))}
    sleep(10)
    TASKS_MSGS = {'msg': 'T4', 'time': str(datetime.today().strftime('%H:%M'))}

@web.route('/loading_status')
@login_required
def loading_status_msgs():
    return TASKS_MSGS



