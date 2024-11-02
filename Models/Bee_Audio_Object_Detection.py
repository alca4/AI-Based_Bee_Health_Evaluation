import os
import glob
import math
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import IPython.display as ipd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, LSTM, Bidirectional, Flatten, Conv1D, MaxPooling1D,
    BatchNormalization, Input, Conv2D, MaxPooling2D, AveragePooling1D, Embedding
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

# Augmentation methods
# Adding white noise
def noise(data):
    noise_amp = 0.05*np.random.uniform()*np.amax(data)   # more noise reduce the value to 0.5
    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
    return data

# Random shifting
def shift(data):
    s_range = int(np.random.uniform(low=-5, high = 5)*1000)  #default at 500
    return np.roll(data, s_range)

# Streching the sound
def stretch(data, rate=0.8):
    data = librosa.effects.time_stretch(data, rate)
    return data

# Pitch tuning
def pitch(data, sample_rate=0.8):
    bins_per_octave = 12
    pitch_pm = 2f
    pitch_change =  pitch_pm * 2*(np.random.uniform())
    data = librosa.effects.pitch_shift(data.astype('float64'),
                                      sample_rate, n_steps=pitch_change,
                                      bins_per_octave=bins_per_octave)
    return data

# Dynamic Change.
def dyn_change(data):
    dyn_change = np.random.uniform(low=-0.5 ,high=7)
    return (data * dyn_change)

# Speed and pitch tuning
def speedNpitch(data):
    length_change = np.random.uniform(low=0.8, high = 1)
    speed_fac = 1.2  / length_change
    tmp = np.interp(np.arange(0,len(data),speed_fac),np.arange(0,len(data)),data)
    minlen = min(data.shape[0], tmp.shape[0])
    data *= 0
    data[0:minlen] = tmp[0:minlen]
    return data

# Confusion matrix plot
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Load and process the training, validation, and test dataset
ref_train = pd.read_csv("./Downloads/Audio_Bee_NoBee_out/Audio_Bee_NoBee_train.csv")
ref_train['sample_name'] = './Downloads/Audio_Bee_NoBee_out/TRAIN/'+ref_train['sample_name'].astype(str)+'.wav'
ref_train.columns = ['path', 'labels']
ref_train.head()

ref_val = pd.read_csv("./Downloads/Audio_Bee_NoBee_out/Audio_Bee_NoBee_val.csv")
ref_val['sample_name'] = './Downloads/Audio_Bee_NoBee_out/VAL/'+ref_val['sample_name'].astype(str)+'.wav'
ref_val.columns = ['path', 'labels']
ref_val.head()

ref_test = pd.read_csv("./Downloads/Audio_Bee_NoBee_out/Audio_Bee_NoBee_test.csv")
ref_test['sample_name'] = './Downloads/Audio_Bee_NoBee_out/TEST/'+ref_test['sample_name'].astype(str)+'.wav'
ref_test.columns = ['path', 'labels']
ref_test.head()

def get_stft(fle):
    # Initialization
    df = pd.DataFrame(columns=['feature'])
    df_noise = pd.DataFrame(columns=['feature'])
    df_speedpitch = pd.DataFrame(columns=['feature'])
    df_shift = pd.DataFrame(columns=['feature'])
    df_stretch = pd.DataFrame(columns=['feature'])
    df_pitch = pd.DataFrame(columns=['feature'])
    df_dyn_change = pd.DataFrame(columns=['feature'])
    cnt = 0

    # Extract feature
    for i in tqdm(fle.path):

        # Load the audio file with specified settings
        X, sample_rate = librosa.load(i
                                      ,res_type='kaiser_fast'
                                      ,duration=10
                                      ,sr=44100
                                      ,offset=0.5
                                    )

        # Calculate signal-to-noise ratio and add noise to the signal
        SNR = 30
        RMS_s=math.sqrt(np.mean(X**2))
        RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
        STD_n = RMS_n
        noise1 = np.random.normal(0, STD_n, X.shape)
        a = X + noise1
        
        # Compute and store STFT
        stft = np.mean(np.abs(librosa.stft(a, n_fft=1024, hop_length=512, win_length=1024, window='hann', center=True, dtype=np.complex64, pad_mode='reflect')),
                        axis=0)
        df.loc[cnt] = [stft]

        # Apply random shifting
        aug = shift(X)
        aug = np.mean(np.abs(librosa.stft(a, n_fft=1024, hop_length=512, win_length=1024, window='hann', center=True, dtype=np.complex64, pad_mode='reflect')),
                      axis=0)
        df_shift.loc[cnt] = [aug]

        # Apply time stretching
        aug = stretch(X)
        aug = np.mean(np.abs(librosa.stft(a, n_fft=1024, hop_length=512, win_length=1024, window='hann', center=True, dtype=np.complex64, pad_mode='reflect')),
                      axis=0)
        df_stretch.loc[cnt] = [aug]

        # Apply pitch shifting
        aug = pitch(X)
        aug = np.mean(np.abs(librosa.stft(a, n_fft=1024, hop_length=512, win_length=1024, window='hann', center=True, dtype=np.complex64, pad_mode='reflect')),
                      axis=0)
        df_pitch.loc[cnt] = [aug]

        # Apply dynamic range change
        aug = dyn_change(X)
        aug = np.mean(np.abs(librosa.stft(a, n_fft=1024, hop_length=512, win_length=1024, window='hann', center=True, dtype=np.complex64, pad_mode='reflect')),
                      axis=0)
        df_dyn_change.loc[cnt] = [aug]

        # Apply noise
        aug = noise(X)
        aug = np.mean(np.abs(librosa.stft(X, n_fft=1024, hop_length=512, win_length=1024, window='hann', center=True, dtype=np.complex64, pad_mode='reflect')),
                      axis=0)
        df_noise.loc[cnt] = [aug]

        # Apply speed and pitch tuning
        aug = speedNpitch(X)
        aug = np.mean(np.abs(librosa.stft(a, n_fft=1024, hop_length=512, win_length=1024, window='hann', center=True, dtype=np.complex64, pad_mode='reflect')),
                      axis=0)
        df_speedpitch.loc[cnt] = [aug]

        cnt += 1

     # Concatenate original and augmented features into separate DataFrames
    df = pd.concat([fle,pd.DataFrame(df['feature'].values.tolist())],axis=1)
    df_noise = pd.concat([fle,pd.DataFrame(df_noise['feature'].values.tolist())],axis=1)
    df_speedpitch = pd.concat([fle,pd.DataFrame(df_speedpitch['feature'].values.tolist())],axis=1)
    df_shift = pd.concat([fle,pd.DataFrame(df_shift['feature'].values.tolist())],axis=1)
    df_stretch = pd.concat([fle,pd.DataFrame(df_stretch['feature'].values.tolist())],axis=1)
    df_pitch = pd.concat([fle,pd.DataFrame(df_pitch['feature'].values.tolist())],axis=1)
    df_dyn_change = pd.concat([fle,pd.DataFrame(df_dyn_change['feature'].values.tolist())],axis=1)

    return df, df_noise, df_speedpitch, df_shift, df_stretch, df_pitch, df_dyn_change

# Generate STFT features for the training, validation, and test dataset
df, df_noise, df_speedpitch, df_shift, df_stretch, df_pitch, df_dyn_change = get_stft(fle=ref_train)
df_val, df_noise_val, df_speedpitch_val, df_shift_val, df_stretch_val, df_pitch_val, df_dyn_change_val = get_stft(fle=ref_val)
df_test, df_noise_test, df_speedpitch_test, df_shift_test, df_stretch_test, df_pitch_test, df_dyn_change_test = get_stft(fle=ref_test)

# Check shapes
df.shape, df_noise.shape, df_speedpitch.shape, df_shift.shape, df_stretch.shape, df_pitch.shape, df_dyn_change.shape

def get_mel(fle):
    # Initialize DataFrames for each type of feature
    df = pd.DataFrame(columns=['feature'])
    df_noise = pd.DataFrame(columns=['feature'])
    df_speedpitch = pd.DataFrame(columns=['feature'])
    df_shift = pd.DataFrame(columns=['feature'])
    df_stretch = pd.DataFrame(columns=['feature'])
    df_pitch = pd.DataFrame(columns=['feature'])
    df_dyn_change = pd.DataFrame(columns=['feature'])
    cnt = 0

    # Loop over each audio file path for feature extraction
    for i in tqdm(fle.path):

        # Load the audio file
        X, sample_rate = librosa.load(i
                            ,res_type='kaiser_fast'
                            ,duration=10
                            ,sr=44100
                            ,offset=0.5)

        # Calculate Mel Spectrogram and normalize
        n_mels=320
        mel_spec = np.mean( librosa.feature.melspectrogram(X, sr=sample_rate, n_mels= n_mels),axis=0 )
        mel_db = (librosa.power_to_db(mel_spec, ref=np.max) + 40)/40
        df.loc[cnt] = [mel_db]

        # Apply random shifting
        aug = shift(X)
        aug = np.mean( librosa.feature.melspectrogram(X, sr=sample_rate, n_mels= n_mels),axis=0 )
        aug = (librosa.power_to_db(aug, ref=np.max) + 40)/40
        df_shift.loc[cnt] = [aug]

        # Apply time stretching
        aug = stretch(X)
        aug = np.mean( librosa.feature.melspectrogram(X, sr=sample_rate, n_mels= n_mels),axis=0 )
        aug = (librosa.power_to_db(aug, ref=np.max) + 40)/40
        df_stretch.loc[cnt] = [aug]

        # Apply pitch shifting
        aug = pitch(X)
        aug = np.mean( librosa.feature.melspectrogram(X, sr=sample_rate, n_mels= n_mels),axis=0 )
        aug = (librosa.power_to_db(aug, ref=np.max) + 40)/40
        df_pitch.loc[cnt] = [aug]

        # Apply dynamic range change
        aug = dyn_change(X)
        aug = np.mean( librosa.feature.melspectrogram(X, sr=sample_rate, n_mels= n_mels),axis=0 )
        aug = (librosa.power_to_db(aug, ref=np.max) + 40)/40
        df_dyn_change.loc[cnt] = [aug]

        # Add noise
        aug = noise(X)
        aug = np.mean( librosa.feature.melspectrogram(X, sr=sample_rate, n_mels= n_mels),axis=0 )
        aug = (librosa.power_to_db(aug, ref=np.max) + 40)/40
        df_noise.loc[cnt] = [aug]

        # Apply speed and pitch tuning
        aug = speedNpitch(X)
        aug = np.mean( librosa.feature.melspectrogram(X, sr=sample_rate, n_mels= n_mels),axis=0 )
        aug = (librosa.power_to_db(aug, ref=np.max) + 40)/40
        df_speedpitch.loc[cnt] = [aug]

        cnt += 1

    # Concatenate original and augmented features into separate DataFrames
    df = pd.concat([fle,pd.DataFrame(df['feature'].values.tolist())],axis=1)
    df_noise = pd.concat([fle,pd.DataFrame(df_noise['feature'].values.tolist())],axis=1)
    df_speedpitch = pd.concat([fle,pd.DataFrame(df_speedpitch['feature'].values.tolist())],axis=1)
    df_shift = pd.concat([fle,pd.DataFrame(df_shift['feature'].values.tolist())],axis=1)
    df_stretch = pd.concat([fle,pd.DataFrame(df_stretch['feature'].values.tolist())],axis=1)
    df_pitch = pd.concat([fle,pd.DataFrame(df_pitch['feature'].values.tolist())],axis=1)
    df_dyn_change = pd.concat([fle,pd.DataFrame(df_dyn_change['feature'].values.tolist())],axis=1)

    return df, df_noise, df_speedpitch, df_shift, df_stretch, df_pitch, df_dyn_change

# Generate Mel spectrogram features for the training, validation and test dataset
df_mel, df_noise_mel, df_speedpitch_mel, df_shift_mel, df_stretch_mel, df_pitch_mel, df_dyn_change_mel = get_mel(fle=ref_train)
df_mel_val, df_noise_mel_val, df_speedpitch_mel_val, df_shift_mel_val, df_stretch_mel_val, df_pitch_mel_val, df_dyn_change_mel_val = get_mel(fle=ref_val)
df_mel_test, df_noise_mel_test, df_speedpitch_mel_test, df_shift_mel_test, df_stretch_mel_test, df_pitch_mel_test, df_dyn_change_mel_test = get_mel(fle=ref_test)

# Check shapes
df_mel_test.shape, df_noise_mel_test.shape, df_speedpitch_mel_test.shape, df_shift_mel_test.shape, df_stretch_mel_test.shape, df_pitch_mel_test.shape, df_dyn_change_mel_test.shape

def get_chroma(fle):

    # Initialize DataFrames for each type of feature
    df = pd.DataFrame(columns=['feature'])
    df_noise = pd.DataFrame(columns=['feature'])
    df_speedpitch = pd.DataFrame(columns=['feature'])
    df_shift = pd.DataFrame(columns=['feature'])
    df_stretch = pd.DataFrame(columns=['feature'])
    df_pitch = pd.DataFrame(columns=['feature'])
    df_dyn_change = pd.DataFrame(columns=['feature'])
    cnt = 0

    # Loop over each audio file path for feature extraction
    for i in tqdm(fle.path):

        # Load the audio file
        X, sr_casvir = librosa.load(i
                                      ,res_type='kaiser_fast'
                                      ,duration=10
                                      ,sr=44100
                                      ,offset=0.5
                                    )

        # Calculate Chroma and store it
        chroma = np.mean(librosa.feature.chroma_stft(X, sr=sr_casvir),
                        axis=0)
        df.loc[cnt] = [chroma]

        # Apply random shifting
        aug = shift(X)
        aug = np.mean(librosa.feature.chroma_stft(X, sr=sr_casvir),
                      axis=0)
        df_shift.loc[cnt] = [aug]

        # Apply time stretching
        aug = stretch(X)
        aug = np.mean(librosa.feature.chroma_stft(X, sr=sr_casvir),
                      axis=0)
        df_stretch.loc[cnt] = [aug]

        # Apply pitch shifting
        aug = pitch(X)
        aug = np.mean(librosa.feature.chroma_stft(X, sr=sr_casvir),
                      axis=0)
        df_pitch.loc[cnt] = [aug]

        # Apply dynamic range change
        aug = dyn_change(X)
        aug = np.mean(librosa.feature.chroma_stft(X, sr=sr_casvir),
                      axis=0)
        df_dyn_change.loc[cnt] = [aug]

        # Add noise
        aug = noise(X)
        aug = np.mean(librosa.feature.chroma_stft(X, sr=sr_casvir),
                      axis=0)
        df_noise.loc[cnt] = [aug]


        # Apply speed and pitch tuning
        aug = speedNpitch(X)
        aug = np.mean(librosa.feature.chroma_stft(X, sr=sr_casvir),
                      axis=0)
        df_speedpitch.loc[cnt] = [aug]

        cnt += 1

    # Concatenate original and augmented features into separate DataFrames
    df = pd.concat([fle,pd.DataFrame(df['feature'].values.tolist())],axis=1)
    df_noise = pd.concat([fle,pd.DataFrame(df_noise['feature'].values.tolist())],axis=1)
    df_speedpitch = pd.concat([fle,pd.DataFrame(df_speedpitch['feature'].values.tolist())],axis=1)
    df_shift = pd.concat([fle,pd.DataFrame(df_shift['feature'].values.tolist())],axis=1)
    df_stretch = pd.concat([fle,pd.DataFrame(df_stretch['feature'].values.tolist())],axis=1)
    df_pitch = pd.concat([fle,pd.DataFrame(df_pitch['feature'].values.tolist())],axis=1)
    df_dyn_change = pd.concat([fle,pd.DataFrame(df_dyn_change['feature'].values.tolist())],axis=1)

    return df, df_noise, df_speedpitch, df_shift, df_stretch, df_pitch, df_dyn_change

# Generate Chroma features for the training, validation and test dataset
df_chroma, df_noise_chroma, df_speedpitch_chroma, df_shift_chroma, df_stretch_chroma, df_pitch_chroma, df_dyn_change_chroma = get_chroma(fle=ref_train)
df_chroma_val, df_noise_chroma_val, df_speedpitch_chroma_val, df_shift_chroma_val, df_stretch_chroma_val, df_pitch_chroma_val, df_dyn_change_chroma_val = get_chroma(fle=ref_val)
df_chroma_test, df_noise_chroma_test, df_speedpitch_chroma_test, df_shift_chroma_test, df_stretch_chroma_test, df_pitch_chroma_test, df_dyn_change_chroma_test = get_chroma(fle=ref_test)

# Check shapes
df_chroma_test.shape, df_noise_chroma_test.shape, df_speedpitch_chroma_test.shape, df_shift_chroma_test.shape, df_stretch_chroma_test.shape, df_pitch_chroma_test.shape, df_dyn_change_chroma_test.shape

def get_mfcc(fle):

    # Initialize DataFrames for each type of feature
    df = pd.DataFrame(columns=['feature'])
    df_noise = pd.DataFrame(columns=['feature'])
    df_speedpitch = pd.DataFrame(columns=['feature'])
    df_shift = pd.DataFrame(columns=['feature'])
    df_stretch = pd.DataFrame(columns=['feature'])
    df_pitch = pd.DataFrame(columns=['feature'])
    df_dyn_change = pd.DataFrame(columns=['feature'])
    cnt = 0

    # Loop over each audio file path for MFCC feature extraction
    for i in tqdm(fle.path):
        # Load the audio file
        X, sample_rate = librosa.load(i
                                      , res_type='kaiser_fast'
                                      ,duration=10
                                      ,sr=44100
                                      ,offset=0.5
                                    )

        # Calculate MFCC feature and store it
        mfccs = np.mean(librosa.feature.mfcc(y=X,
                                            sr=np.array(sample_rate),
                                            n_mfcc=13),
                        axis=0)

        df.loc[cnt] = [mfccs]

        # Apply random shifting
        aug = shift(X)
        aug = np.mean(librosa.feature.mfcc(y=aug,
                                        sr=np.array(sample_rate),
                                        n_mfcc=13),
                      axis=0)
        df_shift.loc[cnt] = [aug]

        # Apply time stretching
        aug = stretch(X)
        aug = np.mean(librosa.feature.mfcc(y=aug,
                                        sr=np.array(sample_rate),
                                        n_mfcc=13),
                      axis=0)
        df_stretch.loc[cnt] = [aug]

        # Apply pitch shifting
        aug = pitch(X)
        aug = np.mean(librosa.feature.mfcc(y=aug,
                                        sr=np.array(sample_rate),
                                        n_mfcc=13),
                      axis=0)
        df_pitch.loc[cnt] = [aug]

        # Apply dynamic range change
        aug = dyn_change(X)
        aug = np.mean(librosa.feature.mfcc(y=aug,
                                        sr=np.array(sample_rate),
                                        n_mfcc=13),
                      axis=0)
        df_dyn_change.loc[cnt] = [aug]

        # Add noise
        aug = noise(X)
        aug = np.mean(librosa.feature.mfcc(y=aug,
                                        sr=np.array(sample_rate),
                                        n_mfcc=13),
                      axis=0)
        df_noise.loc[cnt] = [aug]

        # Apply speed and pitch tuning
        aug = speedNpitch(X)
        aug = np.mean(librosa.feature.mfcc(y=aug,
                                        sr=np.array(sample_rate),
                                        n_mfcc=13),
                      axis=0)
        df_speedpitch.loc[cnt] = [aug]

        cnt += 1

    # Concatenate original and augmented features into separate DataFrames
    df = pd.concat([fle,pd.DataFrame(df['feature'].values.tolist())],axis=1)
    df_noise = pd.concat([fle,pd.DataFrame(df_noise['feature'].values.tolist())],axis=1)
    df_speedpitch = pd.concat([fle,pd.DataFrame(df_speedpitch['feature'].values.tolist())],axis=1)
    df_shift = pd.concat([fle,pd.DataFrame(df_shift['feature'].values.tolist())],axis=1)
    df_stretch = pd.concat([fle,pd.DataFrame(df_stretch['feature'].values.tolist())],axis=1)
    df_pitch = pd.concat([fle,pd.DataFrame(df_pitch['feature'].values.tolist())],axis=1)
    df_dyn_change = pd.concat([fle,pd.DataFrame(df_dyn_change['feature'].values.tolist())],axis=1)

    return df, df_noise, df_speedpitch, df_shift, df_stretch, df_pitch, df_dyn_change

# Generate MFCC features for the training, validation and test dataset
df_mfcc, df_noise_mfcc, df_speedpitch_mfcc, df_shift_mfcc, df_stretch_mfcc, df_pitch_mfcc, df_dyn_change_mfcc = get_mfcc(fle=ref_train)
df_mfcc_val, df_noise_mfcc_val, df_speedpitch_mfcc_val, df_shift_mfcc_val, df_stretch_mfcc_val, df_pitch_mfcc_val, df_dyn_change_mfcc_val = get_mfcc(fle=ref_val)
df_mfcc_test, df_noise_mfcc_test, df_speedpitch_mfcc_test, df_shift_mfcc_test, df_stretch_mfcc_test, df_pitch_mfcc_test, df_dyn_change_mfcc_test = get_mfcc(fle=ref_test)

# Display shapes of each test DataFrame
df_mfcc.shape, df_noise_mfcc.shape, df_speedpitch_mfcc.shape, df_shift_mfcc.shape, df_stretch_mfcc.shape, df_pitch_mfcc.shape, df_dyn_change_mfcc.shape

# Concatenate training DataFrames for final feature sets
df_final = pd.concat([df,df_noise, df_shift, df_pitch],axis=0,sort=False)
df_final=df_final.fillna(0)
df_final.head()

df_mel_final = pd.concat([df_mel,df_noise_mel, df_shift_mel, df_pitch_mel],axis=0,sort=False)
df_mel_final=df_mel_final.fillna(0)
df_mel_final.head()

df_chroma_final = pd.concat([df_chroma, df_noise_chroma, df_shift_chroma, df_pitch_chroma],axis=0,sort=False)
df_chroma_final=df_chroma_final.fillna(0)
df_chroma_final.head()

df_mfcc_final = pd.concat([df_mfcc, df_noise_mfcc, df_shift_mfcc, df_pitch_mfcc],axis=0,sort=False)
df_mfcc_final=df_mfcc_final.fillna(0)
df_mfcc_final.head()

# Display shape of the final MFCC DataFrame
df_mfcc_final.shape

# Concatenate validation DataFrames for final feature sets
df_final_val = pd.concat([df_val])
df_final_val=df_final_val.fillna(0)
df_final_val.head()

df_mel_final_val = pd.concat([df_mel_val])
df_mel_final_val=df_mel_final_val.fillna(0)
df_mel_final_val.head()

df_chroma_final_val = pd.concat([df_chroma_val])
df_chroma_final_val=df_chroma_final_val.fillna(0)
df_chroma_final_val.head()

df_mfcc_final_val = pd.concat([df_mfcc_val])
df_mfcc_final_val=df_mfcc_final_val.fillna(0)
df_mfcc_final_val.head()

# Display shape of the final MFCC validation DataFrame
df_mfcc_final_val.shape

# Concatenate test DataFrames for final feature sets
df_final_test = pd.concat([df_test])
df_final_test=df_final_test.fillna(0)
df_final_test.head()

df_mel_final_test = pd.concat([df_mel_test])
df_mel_final_test=df_mel_final_test.fillna(0)
df_mel_final_test.head()

df_chroma_final_test = pd.concat([df_chroma_test])
df_chroma_final_test=df_chroma_final_test.fillna(0)
df_chroma_final_test.head()

df_mfcc_final_test = pd.concat([df_mfcc_test])
df_mfcc_final_test=df_mfcc_final_test.fillna(0)
df_mfcc_final_test.shape

# Print shapes for training, validation, and test DataFrames for each feature type
print(df_final.shape, df_final_val.shape, df_final_test.shape, df_mel_final.shape, df_mel_final_val.shape, df_mel_final_test.shape, df_chroma_final.shape, df_chroma_final_val.shape, df_chroma_final_test.shape)
print(df_mfcc_final.shape, df_mfcc_final_val.shape, df_mfcc_final_test.shape)

# Extract labels for df_final feature sets
y_train=df_final.labels
y_val=df_final_val.labels
y_test=df_final_test.labels

# Check shapes
y_train.shape, y_val.shape, y_test.shape

def normalize(dat):
    # Drop non-feature columns and normalize data
    dat=(dat.drop(['path','labels'],axis=1))
    mean_train = np.mean(dat, axis=0)
    std_train = np.std(dat, axis=0)

    # Standardize the data
    dat = (dat - mean_train)/std_train

    # Convert to numpy array and count unique values
    dat = np.array(dat)
    unique, counts = np.unique(dat, return_counts=True)
    result = np.column_stack((unique, counts))
    return dat

# Normalize feature sets
df_final_normalized = normalize(df_final)
df_final_val_normalized = normalize(df_final_val)
df_final_test_normalized = normalize(df_final_test)

df_mel_final_normalized = normalize(df_mel_final)
df_mel_final_normalized_val = normalize(df_mel_final_val)
df_mel_final_normalized_test = normalize(df_mel_final_test)

df_chroma_final_normalized = normalize(df_chroma_final)
df_chroma_final_normalized_val = normalize(df_chroma_final_val)
df_chroma_final_normalized_test = normalize(df_chroma_final_test)

df_mfcc_final_normalized = normalize(df_mfcc_final)
df_mfcc_final_normalized_val = normalize(df_mfcc_final_val)
df_mfcc_final_normalized_test = normalize(df_mfcc_final_test)

# One-hot encode the target labels
lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_val = np_utils.to_categorical(lb.fit_transform(y_val))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

# Pickle the LabelEncoder object
filename = 'labels'
outfile = open(filename,'wb')
pickle.dump(lb,outfile)
outfile.close()

# Expand dimensions
# Time-frequency 
df_final_expand = np.expand_dims(df_final_normalized, axis=2)
df_final_val_expand = np.expand_dims(df_final_val_normalized, axis=2)
df_final_test_expand = np.expand_dims(df_final_test_normalized, axis=2)

# Mel spectrogram 
df_mel_final_expand = np.expand_dims(df_mel_final_normalized, axis=2)
df_mel_final_expand_val = np.expand_dims(df_mel_final_normalized_val, axis=2)
df_mel_final_expand_test = np.expand_dims(df_mel_final_normalized_test, axis=2)

# Chroma 
df_chroma_final_expand = np.expand_dims(df_chroma_final_normalized, axis=2)
df_chroma_final_expand_val = np.expand_dims(df_chroma_final_normalized_val, axis=2)
df_chroma_final_expand_test = np.expand_dims(df_chroma_final_normalized_test, axis=2)


# One-hot encode the target labels
lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_val = np_utils.to_categorical(lb.fit_transform(y_val))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

# Pickle the LabelEncoder object
filename = 'labels'
with open(filename, 'wb') as outfile:
    pickle.dump(lb, outfile)

# Expand dimensions 
df_final_expand = np.expand_dims(df_final_normalized, axis=2)
df_final_val_expand = np.expand_dims(df_final_val_normalized, axis=2)
df_final_test_expand = np.expand_dims(df_final_test_normalized, axis=2)

df_mel_final_expand = np.expand_dims(df_mel_final_normalized, axis=2)
df_mel_final_expand_val = np.expand_dims(df_mel_final_normalized_val, axis=2)
df_mel_final_expand_test = np.expand_dims(df_mel_final_normalized_test, axis=2)

df_chroma_final_expand = np.expand_dims(df_chroma_final_normalized, axis=2)
df_chroma_final_expand_val = np.expand_dims(df_chroma_final_normalized_val, axis=2)
df_chroma_final_expand_test = np.expand_dims(df_chroma_final_normalized_test, axis=2)

df_mfcc_final_expand = np.expand_dims(df_mfcc_final_normalized, axis=2)
df_mfcc_final_expand_val = np.expand_dims(df_mfcc_final_normalized_val, axis=2)
df_mfcc_final_expand_test = np.expand_dims(df_mfcc_final_normalized_test, axis=2)

# Stack expanded features to form training, validation, and test sets
X_train = np.dstack((df_final_expand,df_mel_final_expand,df_chroma_final_expand))
X_val = np.dstack((df_final_val_expand,df_mel_final_expand_val,df_chroma_final_expand_val))
X_test = np.dstack((df_final_test_expand,df_mel_final_expand_test,df_chroma_final_expand_test))

# Check shapes
print(X_val.shape)
print(y_val.shape)
print(df_mel_final_normalized.shape)
print(df_mel_final_expand.shape)
print(X_train.shape)


# Model Callbacks
earlystopper = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

# Checkpointer to save the best model
checkpointer = ModelCheckpoint('./Downloads/Audio_Bee_NoBee_out/best_model3.h5'
                               ,monitor='val_accuracy'
                               ,verbose=1
                               ,save_best_only=True
                               ,save_weights_only=True)

# Model Architecture
model = Sequential()

model.add(Conv1D(64, 8, padding='same', input_shape=(df_mfcc_final_normalized.shape[1],1)))
model.add(Activation('relu'))
model.add(Conv1D(64, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(1)))
model.add(Dropout(0.5))

model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(1)))
model.add(Dropout(0.5))

model.add(Conv1D(256, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(256, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(1)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.25))
model.add(Dense(64))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Dropout(0.25))
model.add(Dense(2))
model.add(Activation('softmax'))

# Optimizer and compilation
# opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
# opt = keras.optimizers.Adam(lr=0.0001)
opt = RMSprop(lr=0.00001, decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr = 0.0001), metrics=['accuracy'])
model.summary()

# Batch and step size
batch_size=128
STEP_SIZE_TRAIN= np.ceil(len(X_train)/batch_size)
STEP_SIZE_VALID= np.ceil(len(X_val)/batch_size)

# Model training
def modeling(train_dat, val_dat, model_name):

    # Early stopping and model checkpoint
    earlystopper = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
    checkpointer = ModelCheckpoint(model_name
                                  ,monitor='val_accuracy'
                                  ,verbose=1
                                  ,save_best_only=True
                                  ,save_weights_only=True)

    # Train the model
    model_history=model.fit(
        train_dat
        , y_train
        , batch_size=batch_size
        , epochs=80
        , steps_per_epoch     = STEP_SIZE_TRAIN
        , validation_steps  = STEP_SIZE_VALID
        , validation_data=(val_dat, y_val)
        , verbose=2
        , shuffle=True
        , callbacks=[earlystopper, checkpointer])

# Train the model on different datasets
modeling(df_final_expand, df_final_val_expand, "./Downloads/Audio_Bee_NoBee_out/best_model3_stft.h5")
modeling(df_mel_final_expand, df_mel_final_expand_val, "./Downloads/Audio_Bee_NoBee_out/best_model3_mel.h5")
modeling(df_chroma_final_expand, df_chroma_final_expand_val, "./Downloads/Audio_Bee_NoBee_out/best_model3_chroma.h5")
modeling(df_mfcc_final_expand, df_mfcc_final_expand_val, "./Downloads/Audio_Bee_NoBee_out/best_model3_mfcc.h5")

def perf(dat, y_dat, model_name,preds):
    # Save the model architecture to JSON
    model_json = model.to_json()
    with open("model_json_aug.json", "w") as json_file:
        json_file.write(model_json)

    # Load model architecture from JSON
    json_file = open('model_json_aug.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # Load model weights
    loaded_model.load_weights(model_name)

    # Compile the model 
    opt = Adam (lr=0.0001)
    loaded_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    # Evaluate model performance
    score = loaded_model.evaluate(dat, y_dat, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

    # Generate predictions
    pred_name = loaded_model.predict(dat,
                            batch_size=16,
                            verbose=1)
    pred_name=pred_name.argmax(axis=1)

    # Transform predictions and actual labels back to original form
    pred_name = pred_name.astype(int).flatten()
    pred_name = (lb.inverse_transform((pred_name)))
    pred_name = pd.DataFrame({preds: pred_name})

    # Actual labels
    actual= y_dat.argmax(axis=1)
    actual = actual.astype(int).flatten()
    actual = (lb.inverse_transform((actual)))
    actual = pd.DataFrame({'actualvalues': actual})
    finaldf = actual.join(pred_name)

    return finaldf

finaldf_stft = perf( df_final_expand, y_train, "./Downloads/Audio_Bee_NoBee_out/best_model3_stft.h5", 'Pred_stft')
finaldf_mel = perf( df_mel_final_expand, y_train, "./Downloads/Audio_Bee_NoBee_out/best_model3_mel.h5", 'Pred_mel')
finaldf_chroma = perf( df_chroma_final_expand,y_train, "./Downloads/Audio_Bee_NoBee_out/best_model3_chroma.h5", 'Pred_chroma')
finaldf_mfcc = perf( df_mfcc_final_expand,y_train, "./Downloads/Audio_Bee_NoBee_out/best_model3_mfcc.h5", 'Pred_mfcc')

# Performance evaluation on train, validation, and test sets
finaldf_stft_val = perf( df_final_val_expand, y_val, "./Downloads/Audio_Bee_NoBee_out/best_model3_stft.h5", 'Pred_stft')
finaldf_mel_val = perf( df_mel_final_expand_val,y_val, "./Downloads/Audio_Bee_NoBee_out/best_model3_mel.h5", 'Pred_mel')
finaldf_chroma_val = perf( df_chroma_final_expand_val, y_val, "./Downloads/Audio_Bee_NoBee_out/best_model3_chroma.h5", 'Pred_chroma')
finaldf_mfcc_val = perf( df_mfcc_final_expand_val, y_val, "./Downloads/Audio_Bee_NoBee_out/best_model3_mfcc.h5", 'Pred_mfcc')

finaldf_stft_test = perf( df_final_test_expand, y_test, "./Downloads/Audio_Bee_NoBee_out/best_model3_stft.h5", 'Pred_stft')
finaldf_mel_test = perf( df_mel_final_expand_test, y_test, "./Downloads/Audio_Bee_NoBee_out/best_model3_mel.h5", 'Pred_mel')
finaldf_chroma_test = perf( df_chroma_final_expand_test,y_test, "./Downloads/Audio_Bee_NoBee_out/best_model3_chroma.h5", 'Pred_chroma')
finaldf_mfcc_test = perf( df_mfcc_final_expand_test,y_test, "./Downloads/Audio_Bee_NoBee_out/best_model3_mfcc.h5", 'Pred_mfcc')

# Classification report
def perf(finaldf, pred):

  finaldf['actualvalues']=finaldf['actualvalues'].map(str)
  finaldf['predictedvalues']=finaldf[pred].map(str)

  classes = finaldf.actualvalues.unique()
  classes.sort()
  print(classification_report(finaldf.actualvalues, finaldf.predictedvalues, target_names=classes, digits=4))

# Generate classification reports for each test set
perf(finaldf_stft_test, 'Pred_stft')
perf(finaldf_mel_test, 'Pred_mel')
perf(finaldf_chroma_test, 'Pred_chroma')
perf(finaldf_mfcc_test, 'Pred_mfcc')