import os
import numpy as np
import librosa
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Define constants
SAMPLE_RATE = 22050
DURATION = 4  # in seconds
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)

# Function to load audio files and extract features
def extract_features_from_audio(audio_path):
    signal, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    hop_length = int(len(signal) / 10)  # Calculate hop length based on the signal length
    mfccs = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=13, hop_length=hop_length)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Function to create the neural network model
def create_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Load and preprocess data
positive_samples_dir = r"C:\Users\ksree\PycharmProjects\pythonProject\chainsaw_sounds"
negative_samples_dir = r"C:\Users\ksree\PycharmProjects\pythonProject\non_chainsaw_sounds"

positive_samples = [extract_features_from_audio(os.path.join(positive_samples_dir, file)) for file in
                    os.listdir(positive_samples_dir)]
negative_samples = [extract_features_from_audio(os.path.join(negative_samples_dir, file)) for file in
                    os.listdir(negative_samples_dir)]

X = np.vstack([positive_samples, negative_samples])
y = np.concatenate([np.ones(len(positive_samples)), np.zeros(len(negative_samples))])

# Shuffle data
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Split data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the model
input_shape = X_train[0].shape
model = create_model(input_shape)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Create confusion matrix and classification report
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob >= 0.5).astype(int)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Save the model for later use
model.save('my_model.keras')
