import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from skimage import io, transform
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms, utils, models

sample_submission = './Downloads/Bee_Images_Health_Merge_Class4_Out.csv'

df = pd.read_csv(sample_submission)
df.head()

# Return the count of unique values
df['health'].value_counts()

df['health'] = df['health'].map({"missing": 2,
                                 "pesti": 3,
                                 "ants": 1,
                                 "normal": 0
                                 })

df['health']

df["health"].value_counts()

df.head()

audio_sample_submission = './Downloads/Audio_Bee_Health_Merge1.csv'
# df = dataframe
df_audio = pd.read_csv(audio_sample_submission)
df_audio.head()

df_audio['label'] = df_audio['label'].map({"missing": 2,
                                 "pesti": 3,
                                 "ants": 1,
                                 "healthy": 0
                                 })

df_audio['label'].value_counts()

df_audio['label']

df_audio['label'].value_counts()

df_audio.head()

import os
cpt = sum([len(files) for r, d, files in os.walk('./Downloads/Bee_Images_Health_Merge_Class4_Out3_Class')])
print('output',cpt)

import os
cpt = sum([len(files) for r, d, files in os.walk('./Downloads/Bee_Images_Health_Merge_Class4_Out3_Class/ants/')])
print('output',cpt)
import os
cpt = sum([len(files) for r, d, files in os.walk('./Downloads/Bee_Images_Health_Merge_Class4_Out3_Class/missing/')])
print('output',cpt)
import os
cpt = sum([len(files) for r, d, files in os.walk('./Downloads/Bee_Images_Health_Merge_Class4_Out3_Class/pesti/')])
print('output',cpt)
import os
cpt = sum([len(files) for r, d, files in os.walk('./Downloads/Bee_Images_Health_Merge_Class4_Out3_Class/healthy/')])
print('output',cpt)

import os
cpt = sum([len(files) for r, d, files in os.walk('./Downloads/Bee_Images_Health_Merge_Class')])
print('output',cpt)

import torch
from torch import optim, nn
from torchvision import models, transforms
model = models.vgg16(pretrained=True)

class FeatureExtractor(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor, self).__init__()
		# Extract VGG-16 Feature Layers
    self.features = list(model.features)
    self.features = nn.Sequential(*self.features)
		# Extract VGG-16 Average Pooling Layer
    self.pooling = model.avgpool
		# Convert the image into one-dimensional vector
    self.flatten = nn.Flatten()
		# Extract the first part of fully-connected layer from VGG16
    self.fc = model.classifier[0]

  def forward(self, x):
		# It will take the input 'x' until it returns the feature vector called 'out'
    out = self.features(x)
    out = self.pooling(out)
    out = self.flatten(out)
    out = self.fc(out)
    return out

# Initialize the model
model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model)

# Change the device to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
new_model = new_model.to(device)

from tqdm import tqdm
import numpy as np
import cv2

# Transform the image, so it becomes readable with the model
transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.CenterCrop(512),
  transforms.Resize(448),
  transforms.ToTensor()
])

def get_feature(folder):
    # Will contain the feature
    features = []

    # Iterate each image
    # for i in tqdm(df.file):
    for i in os.listdir(folder):
      # Set the image path
      path = os.path.join(folder, str(i))
      # print(path)
      # Read the file
      img = cv2.imread(path)
      # Transform the image
      img = transform(img)
      # Reshape the image. PyTorch model reads 4-dimensional tensor
      # [batch_size, channels, width, height]
      img = img.reshape(1, 3, 448, 448)
      img = img.to(device)
      # We only extract features, so we don't need gradient
      with torch.no_grad():
        # Extract the feature from the image
        feature = new_model(img)
      # Convert to NumPy Array, Reshape it, and save it to features variable
      features.append(feature.cpu().detach().numpy().reshape(-1))

    # Convert to NumPy Array
    features = np.array(features)
    return features

ants_features=get_feature('./Downloads/Bee_Images_Health_Merge_Class4_Out3_Class/ants/')
missing_features=get_feature('./Downloads/Bee_Images_Health_Merge_Class4_Out3_Class/missing/')
pesti_features=get_feature('./Downloads/Bee_Images_Health_Merge_Class4_Out3_Class/pesti/')
healthy_features=get_feature('./Downloads/Bee_Images_Health_Merge_Class4_Out3_Class/healthy/')

#audio

from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from os import walk
import os

def get_Plots(INPUT_DIR, OUTPUT_DIR):

    ants_wavs = []
    for (_,_,filenames) in walk(INPUT_DIR):
        ants_wavs.extend(filenames)
        break
    ants_wavs

    for ant_wav in ants_wavs:
        # read audio samples
        input_data = read(INPUT_DIR + ant_wav)
        audio = input_data[1]
        # plot the first 1024 samples
        plt.plot(audio)
        # label the axes
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        # set the title
        # plt.title("Sample Wav")
        # display the plot
        plt.savefig(OUTPUT_DIR + ant_wav.split('.')[0] + '.png')
        # plt.show()
        plt.close('all')

get_Plots("./Downloads/Bee_Audio_Health_Merge_4/pesti/", "./Downloads/Bee_Audio_Health_Merge_4/pestiPlots/")
get_Plots("./Downloads/Bee_Audio_Health_Merge_4/ants/", "./Downloads/Bee_Audio_Health_Merge_4/antsPlots/")
get_Plots("./Downloads/Bee_Audio_Health_Merge_4/missing/", "./Downloads/Bee_Audio_Health_Merge_4/missingPlots/")
get_Plots("./Downloads/Bee_Audio_Health_Merge_4/healthy/", "./Downloads/Bee_Audio_Health_Merge_4/healthyPlots/")

from tqdm import tqdm
import numpy as np
import cv2

# Transform the image, so it becomes readable with the model
transform = transforms.Compose([
transforms.ToPILImage(),
transforms.CenterCrop(512),
transforms.Resize(448),
transforms.ToTensor()
])

def get_audio_feature(folder,label):

    # Will contain the feature
    audio_features = []
    y=[]

    # Iterate each image
    # for i in tqdm(df_audio.sample_name):
    for i in os.listdir(folder):
      # Set the image path
      path = os.path.join(folder, str(i))
      # print(path)
      # print(path)
      # Read the file
      img = cv2.imread(path)
      # Transform the image
      img = transform(img)
      # Reshape the image. PyTorch model reads 4-dimensional tensor
      # [batch_size, channels, width, height]
      img = img.reshape(1, 3, 448, 448)
      img = img.to(device)
      # We only extract features, so we don't need gradient
      with torch.no_grad():
        # Extract the feature from the image
        feature = new_model(img)
      # Convert to NumPy Array, Reshape it, and save it to features variable
      audio_features.append(feature.cpu().detach().numpy().reshape(-1))
      y.append(label)

    # Convert to NumPy Array
    audio_features = np.array(audio_features)
    return audio_features, y


ants_audio_features, ants_y=get_audio_feature('./Downloads/Bee_Audio_Health_Merge_4/ants_STFT', 1)
missing_audio_features, missing_y=get_audio_feature('./Downloads/Bee_Audio_Health_Merge_4/missing_STFT/', 2)
pesti_audio_features, pesti_y=get_audio_feature('./Downloads/Bee_Audio_Health_Merge_4/pesti_STFT/', 3)
healthy_audio_features, healthy_y=get_audio_feature('./Downloads/Bee_Audio_Health_Merge_4/healthy_STFT/', 0)

ants_audio_features.shape, missing_audio_features.shape, pesti_audio_features.shape, healthy_audio_features.shape

ants_features.shape,  missing_features.shape, pesti_features.shape, healthy_features.shape

merged_ants_features=np.concatenate((ants_features,ants_audio_features),axis=1)
merged_missing_features=np.concatenate((missing_features,missing_audio_features),axis=1)
merged_pesti_features=np.concatenate((pesti_features,pesti_audio_features),axis=1)
merged_health_features=np.concatenate((healthy_features,healthy_audio_features),axis=1)

merged_ants_features.shape, merged_missing_features.shape,merged_pesti_features.shape, merged_health_features.shape

y=[]
y.extend(ants_y)
y.extend(missing_y)
y.extend(pesti_y)
y.extend(healthy_y)

len(y)

merged_features=np.concatenate((merged_ants_features,merged_missing_features, merged_pesti_features, merged_health_features),axis=0)

merged_features.shape

merged_features.dtype

merged_features

df = pd.DataFrame(merged_features)

df.head()

df['label'] = y

# Normalize the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

X

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=42)

from sklearn.model_selection import train_test_split
X_val,X_test,y_val,y_test = train_test_split(X_test,y_test,test_size = 0.5)

len(X_val)

X_image_train = X_train[:,:4096]
X_image_val = X_val[:,:4096]
X_image_test = X_test[:,:4096]

X_audio_train = X_train[:, -4096:]
X_audio_val = X_val[:, -4096:]
X_audio_test = X_test[:, -4096:]

import keras
from keras.models import Sequential
from keras.layers import Dense

# Neural network
model = Sequential()
model.add(Dense(16, input_dim=8192, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train,validation_data = (X_val,y_val), epochs=20, batch_size=64)

import keras
from keras.models import Sequential
from keras.layers import Dense
# Neural network
model = Sequential()
model.add(Dense(16, input_dim=4096, activation='relu'))

model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_image_train, y_train,validation_data = (X_image_val,y_val), epochs=20, batch_size=64)

###images
y_image_pred = model.predict(X_image_test)
#Converting predictions to label
image_pred = list()
for i in range(len(y_image_pred)):
    image_pred.append(np.argmax(y_image_pred[i]))
#Converting one hot encoded test label to label
image_test = list()
for i in range(len(y_test)):
    image_test.append(np.argmax(y_test[i]))

#images
from sklearn.metrics import accuracy_score
a = accuracy_score(image_pred,image_test)
print('Accuracy is:', a*100)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

print(image_pred)

print(image_test)

from sklearn.metrics import classification_report

classes = ['healthy', 'ant problems', 'missing queen','pesticide']
print(classification_report(image_test, image_pred,  target_names=classes, digits=4))

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


classes = ['healthy', 'ant problem', 'missing queen','pesticide']

cm = confusion_matrix(y_true = image_test,y_pred = image_pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

disp.plot(cmap=plt.cm.Blues)
plt.show()

#audio
history = model.fit(X_audio_train, y_train,validation_data = (X_audio_val,y_val), epochs=20, batch_size=64)

#audio
y_audio_pred = model.predict(X_audio_test)
#Converting predictions to label
audio_pred = list()
for i in range(len(y_audio_pred)):
   audio_pred.append(np.argmax(y_audio_pred[i]))
#Converting one hot encoded test label to label
audio_test = list()
for i in range(len(y_test)):
   audio_test.append(np.argmax(y_test[i]))

#audio
from sklearn.metrics import accuracy_score
a = accuracy_score(audio_pred,audio_test)
print('Accuracy is:', a*100)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

print(audio_pred)

print(audio_test)

from sklearn.metrics import classification_report

classes = ['healthy', 'ant problems', 'missing queen','pesticide']
print(classification_report(audio_test, audio_pred,  target_names=classes, digits=4))

from sklearn.metrics import classification_report

classes = ['healthy', 'ant problems', 'missing queen','pesticide']
print(classification_report(audio_pred, audio_test,  target_names=classes, digits=4))

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


classes = ['healthy', 'ant problem', 'missing queen','pesticide']

cm = confusion_matrix(y_true = audio_test,y_pred = audio_pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

disp.plot(cmap=plt.cm.Blues)
plt.show()

output=df['label'].values

output

# visual & audio
y_pred = model.predict(X_test)
#Convert predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Convert one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

print(pred)

print(test)

from sklearn.metrics import classification_report

classes = ['healthy', 'ant problems', 'missing queen','pesticide']
print(classification_report(pred, test, target_names=classes, digits=4))

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


classes = ['healthy', 'ant problem', 'missing queen','pesticide']

cm = confusion_matrix(y_true = test,y_pred = pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

disp.plot(cmap=plt.cm.Blues)
plt.show()