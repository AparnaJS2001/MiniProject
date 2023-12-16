from flask import Flask, render_template, request, jsonify
import numpy as np
from main import extract_features_from_audio, model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    # Receive the audio file from the frontend
    audio_blob = request.files['audio'].read()

    # Extract features from the audio
    new_audio_features = extract_features_from_audio(audio_blob)

    # Make prediction using the model
    prediction = model.predict(np.expand_dims(new_audio_features, axis=0))[0, 0]
    threshold = 0.5

    if prediction >= threshold:
        result = "Chainsaw sound detected!"
    else:
        result = "No chainsaw sound detected."

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
