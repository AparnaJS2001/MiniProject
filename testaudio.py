import numpy as np

from main import extract_features_from_audio, model

new_audio_path = r'C:\Users\ksree\PycharmProjects\pythonProject\chainsaw_sounds\0 (4).wav'
new_audio_features = extract_features_from_audio(new_audio_path)

prediction = model.predict(np.expand_dims(new_audio_features, axis=0))[0, 0]
print("prediction value:",prediction)
threshold = 0.5 # Adjust this threshold as needed

if prediction >= threshold:
    print("Chainsaw sound detected!")
else:
    print("No chainsaw sound detected.")
