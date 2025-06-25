import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from streamlit_audiorec import st_audiorec
import io

# Load pre-trained model
model = tf.keras.models.load_model('emotion_model.h5')

# Feature extraction
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    return np.vstack([mfcc, mel, chroma, contrast, tonnetz]).T

# Audio preprocessing
def preprocess_audio(y):
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    target_length = int(22050 * 3.5)
    if len(y_trimmed) < target_length:
        y_padded = np.pad(y_trimmed, (0, target_length - len(y_trimmed)))
    else:
        y_padded = y_trimmed[:target_length]
    return y_padded

def main():
    st.title("🎤 Speech Emotion Recognition")

    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    audio_data = st_audiorec()

    if audio_file or audio_data is not None:
        if audio_file:
            audio_bytes = audio_file.read()
        else:
            audio_bytes = audio_data

        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        y_processed = preprocess_audio(y)
        features = extract_features(y_processed, sr)
        features = np.expand_dims(features, axis=0)

        prediction = model.predict(features)
        emotion_idx = np.argmax(prediction)
        emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad']
        emotion = emotions[emotion_idx]

        st.subheader("Results")
        st.write(f"**Predicted Emotion:** {emotion}")
        st.bar_chart(dict(zip(emotions, prediction[0])))

        st.audio(audio_bytes, format='audio/wav')

if __name__ == "__main__":
    main()
