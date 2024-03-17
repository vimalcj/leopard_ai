from flask import (Flask, redirect, render_template, request,jsonify,
                   send_from_directory, url_for)
import librosa
import numpy as np
import tempfile
import os

app = Flask(__name__)

@app.route('/hello')
def index():
   print('Health is good')
   return jsonify("Health is good")

def extract_mean(file):
    y, sr = librosa.load(file)
    # Extract audio features
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    # Print or use the extracted features
    meansList = []
    # Print or use the extracted features
    meansList.append(chroma_stft)
    # print("Chroma STFT:",np.mean(chroma_stft))
    # print("RMS:", rms)
    meansList.append(rms)
    # print("Spectral Centroid:", spectral_centroid)
    meansList.append(spectral_centroid)
    # print("Spectral Bandwidth:", spectral_bandwidth)
    meansList.append(spectral_bandwidth)
    # print("Rolloff:", rolloff)
    meansList.append(rolloff)
    # print("Zero Crossing Rate:", zero_crossing_rate)
    meansList.append(zero_crossing_rate)
    length = len(mfccs)

    for i in range(length):
        # print("mfccs"+str(i)+":"+str(np.mean(mfccs[i])))
        meansList.append(np.mean(mfccs[i]))
    #print(dataAll)
    return {
	    'data': [*meansList]
	}

@app.route('/extract_audio_features', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded file to a temporary location
        _, temp_audio_file = tempfile.mkstemp(suffix='.wav')
        file.save(temp_audio_file)

        # Extract audio features
        audio_features = extract_mean(temp_audio_file)

        # Delete the temporary audio file
        os.remove(temp_audio_file)

        return jsonify(audio_features)

if __name__ == '__main__':
   app.run()
