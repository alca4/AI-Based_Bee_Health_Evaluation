import pandas as pd
ref = pd.read_csv("./Downloads/Audio_Bee_Health_Merge.csv")

ref['sample_name'] = ref['sample_name'].astype(str)+'.wav'

ref.columns = ['path', 'labels']
ref.head()

image_label=dict(zip(ref.path, ref.labels))

#--------feature extraction tools----------#
#stft
def stft_extraction(filepath, n_chunks):
  x, sr = librosa.load(filepath)
  s = np.abs(librosa.stft(x, n_fft=1024, hop_length=512, win_length=1024, window='hann',
                           center=True, dtype=np.complex64, pad_mode='reflect'))

  summ_s = mean(s, n_chunks)
  return summ_s

#complex stft - using scipy.stft
def complex_stft(filepath, n_chunks):
    x, fs = librosa.load(filepath)
    zs = np.abs(librosa.stft(x, n_fft=1024, hop_length=512, win_length=1024, window='hann',
                center=True, dtype=np.complex64, pad_mode='reflect'))

    real = zs.real
    imag = zs.imag

    summ_real = mean(real, n_chunks)
    summ_imag = mean(imag, n_chunks)
    summ_complex = summ_real**2 + summ_imag**2
    return summ_complex

#cqt
def cqt_extraction(filepath, n_chunks):
    x, sr = librosa.load(filepath)
    cqt = np.abs(librosa.cqt(x, sr=sr, n_bins=513, bins_per_octave=216))
    summ_cqt = mean(cqt, n_chunks)
    return summ_cqt

#mfccs - as a baseline
def mfccs_extraction(filepath):
  x, sr = librosa.load(filepath)
  mfccs = librosa.feature.mfcc(x, n_mfcc=20, sr=sr)
  return mfccs

#STFT without mean-spectrogram
def stft_classic(filepath):
    x, sr = librosa.load(filepath)
    s = np.abs(librosa.stft(x, n_fft=1024, hop_length=512, win_length=1024, window='hann',
                           center=True, dtype=np.complex64, pad_mode='reflect'))
    return s

#CQT without mean-spectrogram
def  cqt_classic(filepath):
    x, sr = librosa.load(filepath)
    cqt = np.abs(librosa.cqt(x, sr=sr, n_bins=513, bins_per_octave=216))
    return cqt

#------------approach selection------------------#
def feature_extraction(filepath, n_chunks, mode):

  if mode == 0:
      s = stft_extraction(filepath, n_chunks)
  elif mode == 1:
      s = complex_stft(filepath, n_chunks)
  elif mode == 2:
      s = cqt_extraction(filepath, n_chunks)
  elif mode == 3:
      s = mfccs_extraction(filepath)
  elif mode == 4:
      s = stft_classic(filepath)
  elif mode == 5:
      s = cqt_classic(filepath)
  return s

def mean(s, n_chunks):
    m, f = s.shape
    mod = m % n_chunks
    #print(mod)
    if m % n_chunks != 0:
        s = np.delete(s, np.s_[0:mod] , 0)
    stft_mean = []
    split = np.split(s, n_chunks, axis = 0)
    for i in range(0, n_chunks):
        stft_mean.append(split[i].mean(axis=0))
    stft_mean = np.asarray(stft_mean)
    return stft_mean

import os

import librosa
import math
import numpy as np
from pathlib import Path

dir='./Downloads/Bee_Audio_Health_Merge/'
n_chunks = 16
features_q = []
queen_q = []
#mode = 3 #could not broadcast input array from shape (20,432) into shape (20,)
#mode = 4 #could not broadcast input array from shape (513, 431) into shape (513,)
#mode = 1 #shape (16,432) into shape (16,)
#mode = 2 #shape (16,432) into shape (16,) #UserWarning: n_fft=8192 is too small for input signal of length=3446
mode = 0 #shape 16
#mode = 5 #shape 513

for filename in os.listdir(dir):
    # if 'exp' in filename:
    if filename.endswith(".wav"):
        filepath = os.path.join(dir, filename)
        # out = feature_extraction(filepath, n_chunks, mode)
        out = feature_extraction(filepath, n_chunks, mode)

        #16 needed to be changed
        if out.shape == (16,431):
                # print('len',len(out))
                #16 needed to be changed
                a = np.zeros([16,1], dtype = int)
                out=np.append(out, a, axis=1)

        # print(out.shape)
        features_q.append(out)
        queen_q.append(image_label[filename])


# print('features_q', features_q)
print('out', out.shape)
features_q = np.asarray(features_q)
queen_q = np.asarray(queen_q)

print('features_q', features_q.shape)
print('queen_q', queen_q.shape)

import os
from sklearn.model_selection import train_test_split
import numpy as np
random_state=42

X_train, X_test, Y_train, Y_test = train_test_split(features_q, queen_q, test_size=0.2, random_state=random_state)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=random_state)

print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape)

import numpy as np
import matplotlib.pyplot as plt
import os
import re
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D , MaxPooling2D
from keras.layers import ELU, PReLU, LeakyReLU

from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

n_outputs=4
model=Sequential()
model.add(Conv2D(16, kernel_size=(3,3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1), padding='same'))

model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))

model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))

model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3,1), activation='relu', padding='same'))

model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))


model.add(Flatten())


model.add(Dense(32 , activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization,LSTM,Reshape,Input, Lambda,Bidirectional

import numpy as np
import matplotlib.pyplot as plt
import os
import re
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D , MaxPooling2D
from keras.layers import ELU, PReLU, LeakyReLU

from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

n_outputs=4

model_LSTM = Sequential()
model_LSTM.add(LSTM(128, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))

model_LSTM.add(Dense(64, activation='relu'))
model_LSTM.add(Dropout(0.4))

model_LSTM.add(Dense(32, activation='relu'))
model_LSTM.add(Dropout(0.4))

model_LSTM.add(Dense(4))
model_LSTM.add(Activation('softmax'))
model_LSTM.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

print(X_train.shape[0], X_train.shape[1], X_train.shape[2])

X_train = X_train.reshape(-1,  X_train.shape[1], X_train.shape[2], 1)
X_val = X_val.reshape(-1, X_val.shape[1], X_val.shape[2],  1)
X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2],  1)

Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)
Y_val = Y_val.reshape(-1, 1)
print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape, Y_test.shape)

le = LabelEncoder()
Y_train = to_categorical(le.fit_transform(Y_train))
Y_val = to_categorical(le.fit_transform(Y_val))
Y_test = to_categorical(le.fit_transform(Y_test))

#Training the network

import tensorflow as tf
from keras.layers import Conv2D , MaxPooling2D
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau

earlystopper = EarlyStopping(monitor='val_accuracy', patience=20, verbose=1)
# Save the best model during the traning
checkpointer = ModelCheckpoint('audio_best_model_cnn.h5'
                               ,monitor='val_accuracy'
                               ,verbose=1
                               ,save_best_only=True
                               ,save_weights_only=True)

batch_size = 128
epochs = 15
start = datetime.now()
model_history=model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                validation_data=(X_val, Y_val), verbose=1, shuffle=True,callbacks=[earlystopper, checkpointer])
duration = datetime.now() - start
print("Training completed in time: ", duration)

score_test = model.evaluate(X_test, Y_test, verbose=1)
print("Testing Accuracy: ", score_test[1])

score_test

Y_pred = model.predict(X_test, batch_size=16, verbose=1)

Y_pred = np.argmax(np.round(Y_pred), axis=1)

Y_pred

actual=Y_test.argmax(axis=1)

actual

classes = ['active', 'missing_queen','ant_problems','pesticide']

print(classification_report(Y_pred, actual, target_names=classes, digits=4))

#Training the network

import tensorflow as tf
from keras.layers import Conv2D , MaxPooling2D
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau


earlystopper = EarlyStopping(monitor='val_accuracy', patience=20, verbose=1)

checkpointer = ModelCheckpoint('best_model_LSTM.h5'
                               ,monitor='val_accuracy'
                               ,verbose=1
                               ,save_best_only=True
                               ,save_weights_only=True)

batch_size = 128

start = datetime.now()
model_history=model_LSTM.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                validation_data=(X_val, Y_val), verbose=1, shuffle=True,callbacks=[earlystopper, checkpointer])
duration = datetime.now() - start
print("Training completed in time: ", duration)

#predicting
Y_pred_LSTM = model_LSTM.predict(X_test, batch_size=16, verbose=1)

Y_pred_LSTM = np.argmax(np.round(Y_pred_LSTM), axis=1)

actual=Y_test.argmax(axis=1)

classes = ['active', 'missing_queen','ant_problems','pesticide']
print(classification_report(Y_pred_LSTM, actual, target_names=classes, digits=4))