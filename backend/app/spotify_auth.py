import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
from dotenv import get_key
import uuid
import pathlib
import os
from flask import Blueprint, request, jsonify
import hashlib
import pandas as pd
import time

# Gets the /backend absolute path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV_PATH = os.path.join(PROJECT_ROOT, '.env')

ADMIN_SPOTOBJ = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=get_key(ENV_PATH, 'SPOTIPY_CLIENT_ID'),
    client_secret=get_key(ENV_PATH, 'SPOTIPY_CLIENT_SECRET')
))

SP_AUTH = SpotifyOAuth(
    client_id=get_key(ENV_PATH, 'SPOTIPY_CLIENT_ID'),
    client_secret=get_key(ENV_PATH, 'SPOTIPY_CLIENT_SECRET'),
    redirect_uri='http://127.0.0.1:5173/auth/callback'
)

def get_hashed_session(user_token):
    # Uses a hash of the spotify id of the user as the session id
    user_id = spotipy.Spotify(auth=user_token['access_token']).current_user()['id']
    session_salt = get_key(ENV_PATH, 'SALT_AND_PEPPER')
    user_session_id = hashlib.sha256((user_id + session_salt).encode()).hexdigest()
    return user_session_id

SESSIONS = pd.DataFrame(columns=['token', 'trusted_ips'])

spotify_auth = Blueprint('spotify_auth', __name__)

@spotify_auth.route('/userdata', methods=['POST'])
def get_user_data():
    global SESSIONS, SP_AUTH
    accessCode = request.json.get('code')
    try:
        user_token = SP_AUTH.get_access_token(code=accessCode, as_dict=True)
        user_token['expires_in'] = time.time() + user_token['expires_in']
    except Exception as e:
        return jsonify({'access_granted':False, 'message':e.__str__()})
    
    user_session_id = get_hashed_session(user_token=user_token)

    if user_session_id in SESSIONS.index:
        SESSIONS.at[user_session_id, 'token'] = user_token
    else:
        SESSIONS.loc[user_session_id] = {'token':user_token}

    return jsonify({ 'access_granted':True, 'message':user_session_id})

@spotify_auth.route('/search', methods=['POST'])
def get_search():
    global ADMIN_SPOTOBJ
    query = request.json.get('query')
    search_results = ADMIN_SPOTOBJ.search(query)
    tracks = []
    for item in search_results['tracks']['items']:
        tracks.append({
            'id': item['id'],
            'name': item['name'],
            'images': item['album']['images'][0]['url'] if item['album']['images'] else '',
            'artists': [i['name'] for i in item['artists']],
            'album': item['album']['name']
        })
    return jsonify(tracks)

@spotify_auth.route('/userdata/profile', methods=['POST'])
def get_user_profile_data():
    global SESSIONS
    user_session_id = request.json.get('session_id')

    if user_session_id in SESSIONS.index:
        user_token = SESSIONS.at[user_session_id, 'token']
    else:
        return jsonify({'error':'Invalid Session ID'})
    
    if not user_token:
        return jsonify({'error':'Not Authorized'})

    # Refresh the token if we have one that's expired
    if user_token['expires_in'] < time.time():
        user_token = SP_AUTH.refresh_access_token(user_token['refresh_token'])
        user_token['expires_in'] = time.time() + user_token['expires_in']

    user_instance = spotipy.Spotify(auth=user_token['access_token'])

    profile_info = user_instance.current_user()

    return jsonify({
        'display_name':profile_info['display_name'],
        'profile_photo':{
            'url':profile_info['images'][0]['url'],
            'height':profile_info['images'][0]['height'],
            'width':profile_info['images'][0]['width']
        },
    })
    