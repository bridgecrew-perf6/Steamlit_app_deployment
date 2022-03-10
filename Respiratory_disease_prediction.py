import streamlit as st
import pandas as pd
import numpy as np

import librosa as lb
import librosa.display as lbd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme(style="whitegrid")
sns.set(rc={'figure.figsize':(12,7)})

import librosa
import librosa.display
import soundfile as sf

import sklearn

import os

import tensorflow as tf
from tensorflow import keras

st.title('Respiratory Disease Prediction using Deep Learning')

uploaded_audio_file = st.file_uploader("Choose the audio file for prediction")

def get_all_features(f_audio_file):
    signal, sample_rate=lb.load(f_audio_file, sr=22050)

    mfcc=librosa.feature.mfcc(signal,sr=sample_rate)
    chroma_stft=librosa.feature.chroma_stft(signal,sr=sample_rate)
    mel_spec=librosa.feature.melspectrogram(signal,sr=sample_rate)
    spectral_centroids = librosa.feature.spectral_centroid(signal, sr=sample_rate)
    spectral_rolloff = librosa.feature.spectral_rolloff(signal, sr=sample_rate)

    return mfcc,chroma_stft,mel_spec,spectral_centroids,spectral_rolloff

def final_feature_metric(uploaded_audio_file):
    mfcc = []
    chroma_stft = []
    mel_spec = []
    spectral_centroids = []
    spectral_rolloff = []

    for index, record in tqdm_notebook(X_train.iterrows()):
        f_audio_file = uploaded_audio_file

        mfcc_loc = get_all_features(f_audio_file)[0]
        chroma_stft_loc = get_all_features(f_audio_file)[1]
        mel_spec_loc = get_all_features(f_audio_file)[2]
        spectral_centroids_loc = get_all_features(f_audio_file)[3]
        spectral_rolloff_loc = get_all_features(f_audio_file)[4]

        mfcc.append(mfcc_loc)
        chroma_stft.append(chroma_stft_loc)
        mel_spec.append(mel_spec_loc)
        spectral_centroids.append(spectral_centroids_loc)
        spectral_rolloff.append(spectral_rolloff_loc)

    mfcc_final = np.array(mfcc)
    chroma_stft_final = np.array(chroma_stft)
    mel_spec_final = np.array(mel_spec)
    spectral_centroids_final = np.array(spectral_centroids)
    spectral_rolloff_final = np.array(spectral_rolloff)

    return mfcc_final,chroma_stft_final,mel_spec_final,spectral_centroids_final,spectral_rolloff_final

if uploaded_audio_file is not None:
    
    body = 'The Audio file has been successfully uploaded. Please wait while we are processing the audio file !!!!'
    st.subheader(body, anchor=None)
    
    signal, sample_rate = librosa.load(uploaded_audio_file, sr=22050)

    mfcc = final_feature_metric[0]
    chroma_stft = final_feature_metric[1]
    mel_spec = final_feature_metric[2]
    spectral_centroids = final_feature_metric[3]
    spectral_rolloff = final_feature_metric[4]

    new_model = tf.keras.models.load_model('drive/myDrive/Deep_learning_model/my_final_model.h5')

    disease_prediction = net.predict({"mfcc": mfcc,
                  "croma": chroma_stft,
                  "mspec": mel_spec_test,
                  "spectral_centroids": spectral_centroids,
                  "spectral_rolloff": spectral_rolloff})

    prediction_txt = 'The Model Predicted the Patient has ' + disease_prediction + 'Please consult a doctor for the same!'
    st.subheader(body, anchor=None)
    
    # Printing the spectrum of the audio file
    spectrum = np.fft.fft(signal)
    mag = np.abs(spectrum)
    freq = np.linspace(0, sample_rate, len(mag))
    
    des_1 = 'The Spectrum of the Audio file'
    st.subheader(des_1, anchor=None)
    fig, ax = plt.subplots()
    ax.plot(freq, mag)

    st.pyplot(fig)
    
    # Converting the above wav file to a Short Time Fourier Transform or Spectrogram

    # Window considering when doing a fft
    num_sample_per_fft = 2048
    
    # Amount shifting fft to the right
    hop_len = 512
    
    stft = librosa.core.stft(signal, hop_length=hop_len, n_fft=num_sample_per_fft)
    spectrogram = np.abs(stft)
    
    # Smothening the Spectrogram
    log_of_spectrogram = librosa.amplitude_to_db(spectrogram)
    
    des_2 = 'The Spectogram of the Audio file'
    st.subheader(des_2, anchor=None)
    fig2, ax2 = plt.subplots()
    img = librosa.display.specshow(log_of_spectrogram, sr=sample_rate, hop_length=hop_len, ax=ax2)

    fig2.colorbar(img, ax=ax2)
    
    st.pyplot(fig2)
    
    # Calculating MFCCs
    mfcc = librosa.feature.mfcc(signal, n_fft=num_sample_per_fft, hop_length=hop_len, n_mfcc=13)
    
    des_3 = 'The MFCC of the Audio file'
    st.subheader(des_3, anchor=None)
    fig3, ax3 = plt.subplots()
    
    img = librosa.display.specshow(mfcc, sr=sample_rate, hop_length=hop_len, ax=ax3)

    fig3.colorbar(img, ax=ax3)
    
    st.pyplot(fig3)
    
    def min_max_normalize(signal, axis=0):
        return sklearn.preprocessing.minmax_scale(signal, axis=axis)
    
    # Calculating Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(signal, sr=sample_rate)[0]
    #norm_spectral_centroids = min_max_normalize(spectral_centroids)
    
    cen_frames = range(len(spectral_centroids))
    time_frame = librosa.frames_to_time(cen_frames)
    
    des_4 = 'The Spectral Centroids of the Audio file'
    st.subheader(des_4, anchor=None)
    fig4, ax4 = plt.subplots()
    
    img = librosa.display.waveshow(signal* 50, sr=sample_rate, alpha=0.4, ax=ax4)
    ax4.plot(time_frame, min_max_normalize(spectral_centroids), color='r')
    
    st.pyplot(fig4)
    
    


