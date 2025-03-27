from flask import Flask
from flask_cors import CORS
from app.data_pipeline import data_pipeline
from app.spotify_auth import spotify_auth
import time
import traceback

app = Flask(__name__)

app.register_blueprint(data_pipeline)
app.register_blueprint(spotify_auth)
CORS(app)   # Allow frontend to connect

if __name__ == '__main__':
    while True:
        try:
            app.run(host='0.0.0.0', port=5000, debug=True)
        except Exception as e:
            print("Server crashed or disconnected. Attempting restart in 5 seconds...")
            traceback.print_exc()
            time.sleep(5)