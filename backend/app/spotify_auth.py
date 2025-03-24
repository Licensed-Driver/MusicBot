import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import get_key
import pathlib
import os
from flask import Blueprint, request, jsonify

# Gets the /backend absolute path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV_PATH = os.path.join(PROJECT_ROOT, '.env')

spotObj = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=get_key(ENV_PATH, 'SPOTIPY_CLIENT_ID'),
    client_secret=get_key(ENV_PATH, 'SPOTIPY_CLIENT_SECRET')
))

spotify_auth = Blueprint('spotify_auth', __name__)

@spotify_auth.route('/search', methods=['POST'])
def get_search():
    query = request.json.get('query')
    search_results = spotObj.search(query)
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