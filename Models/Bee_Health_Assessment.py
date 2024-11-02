import os
from os import walk
import copy
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from skimage import io, transform
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms, utils, models
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense
from scipy.io.wavfile import read
from tqdm import tqdm

# Load and display the initial data for bee image health
bee_sample = './Downloads/Bee_Images_Health_Merge_Class4_Out.csv'
df = pd.read_csv(bee_sample)
df.head()

# Map and count unique values in 'health' column
df['health'].value_counts()
df['health'] = df['health'].map({
    "missing": 2,
    "pesti": 3,
    "ants": 1,
    "normal": 0
})
df['health']
df["health"].value_counts()
df.head()

# Load and display the initial data for audio bee health
audio_bee_sample = './Downloads/Audio_Bee_Health_Merge1.csv'
df_audio = pd.read_csv(audio_bee_sample)
df_audio.head()

# Map and count unique values in 'label' column for audio data
df_audio['label'] = df_audio['label'].map({
    "missing": 2,
    "pesti": 3,
    "ants": 1,
    "healthy": 0
})
df_audio['label'].value_counts()
df_audio['label']
df_audio['label'].value_counts()
df_audio.head()

# Define a FeatureExtractor class based on VGG-16
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
	# Forward pass: process input 'x' and return feature vector 'out'
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

# Define image transformation pipeline for model compatibility
transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.CenterCrop(512),
  transforms.Resize(448),
  transforms.ToTensor()
])

def get_feature(folder):
    # List to store extracted features
    features = []

    # Iterate through each image in the folder
    for i in os.listdir(folder):
      # Set the image path
      path = os.path.join(folder, str(i))

      # Read and transform the image
      img = cv2.imread(path)
      img = transform(img)

      # Reshape the image for PyTorch model compatibility: [batch_size, channels, width, height]
      img = img.reshape(1, 3, 448, 448)
      img = img.to(device)

      # Extract features without computing gradients
      with torch.no_grad():
        feature = new_model(img)

      # Convert feature to NumPy array, reshape, and add to features list
      features.append(feature.cpu().detach().numpy().reshape(-1))

    # Convert features list to a NumPy array and return
    features = np.array(features)
    return features

# Extract features for each class of bee images
ants_features=get_feature('./Downloads/Bee_Images_Health_Merge_Class4_Out3_Class/ants/')
missing_features=get_feature('./Downloads/Bee_Images_Health_Merge_Class4_Out3_Class/missing/')
pesti_features=get_feature('./Downloads/Bee_Images_Health_Merge_Class4_Out3_Class/pesti/')
healthy_features=get_feature('./Downloads/Bee_Images_Health_Merge_Class4_Out3_Class/healthy/')

def get_Plots(INPUT_DIR, OUTPUT_DIR):
    # Gather all audio filenames in the input directory
    ants_wavs = []
    for (_,_,filenames) in walk(INPUT_DIR):
        ants_wavs.extend(filenames)
        break

    # Process each audio file
    for ant_wav in ants_wavs:
        # Read audio samples
        input_data = read(INPUT_DIR + ant_wav)
        audio = input_data[1]

        # Plot the audio waveform
        plt.plot(audio)
        plt.ylabel("Amplitude")
        plt.xlabel("Time")

        # Save the plot as an image in the output directory
        plt.savefig(OUTPUT_DIR + ant_wav.split('.')[0] + '.png')
        plt.close('all')

# Generate and save plots for each category of bee audio
get_Plots("./Downloads/Bee_Audio_Health_Merge_4/pesti/", "./Downloads/Bee_Audio_Health_Merge_4/pestiPlots/")
get_Plots("./Downloads/Bee_Audio_Health_Merge_4/ants/", "./Downloads/Bee_Audio_Health_Merge_4/antsPlots/")
get_Plots("./Downloads/Bee_Audio_Health_Merge_4/missing/", "./Downloads/Bee_Audio_Health_Merge_4/missingPlots/")
get_Plots("./Downloads/Bee_Audio_Health_Merge_4/healthy/", "./Downloads/Bee_Audio_Health_Merge_4/healthyPlots/")

def get_audio_feature(folder,label):
    # List to store extracted audio features and labels
    audio_features = []
    y=[]

    # Iterate through each file in the folder
    for i in os.listdir(folder):
      # Set the file path
      path = os.path.join(folder, str(i))

      # Read and transform the image
      img = cv2.imread(path)
      img = transform(img)

      # Reshape the image. PyTorch model reads 4-dimensional tensor
      # [batch_size, channels, width, height]
      img = img.reshape(1, 3, 448, 448)
      img = img.to(device)

      # Extract features without computing gradients
      with torch.no_grad():
        feature = new_model(img)
    
      # Append the extracted features and labels
      audio_features.append(feature.cpu().detach().numpy().reshape(-1))
      y.append(label)

    # Convert to NumPy array for output
    audio_features = np.array(audio_features)
    return audio_features, y

# Extract audio features and labels for each category
ants_audio_features, ants_y=get_audio_feature('./Downloads/Bee_Audio_Health_Merge_4/ants_STFT', 1)
missing_audio_features, missing_y=get_audio_feature('./Downloads/Bee_Audio_Health_Merge_4/missing_STFT/', 2)
pesti_audio_features, pesti_y=get_audio_feature('./Downloads/Bee_Audio_Health_Merge_4/pesti_STFT/', 3)
healthy_audio_features, healthy_y=get_audio_feature('./Downloads/Bee_Audio_Health_Merge_4/healthy_STFT/', 0)

# Print the shape of the audio feature arrays
print("Audio Feature Shapes:")
print(f"Ants: {ants_audio_features.shape}, Missing: {missing_audio_features.shape}, "
      f"Pesti: {pesti_audio_features.shape}, Healthy: {healthy_audio_features.shape}")

# Print the shape of the image feature arrays
print("Image Feature Shapes:")
print(f"Ants: {ants_features.shape}, Missing: {missing_features.shape}, "
      f"Pesti: {pesti_features.shape}, Healthy: {healthy_features.shape}")

# Merge audio and image features for each category
merged_ants_features=np.concatenate((ants_features,ants_audio_features),axis=1)
merged_missing_features=np.concatenate((missing_features,missing_audio_features),axis=1)
merged_pesti_features=np.concatenate((pesti_features,pesti_audio_features),axis=1)
merged_health_features=np.concatenate((healthy_features,healthy_audio_features),axis=1)

# Print the shape of the merged feature arrays
print("Merged Feature Shapes:")
print(f"Ants: {merged_ants_features.shape}, Missing: {merged_missing_features.shape}, "
      f"Pesti: {merged_pesti_features.shape}, Healthy: {merged_health_features.shape}")

# Combine labels from each category
y = []
y.extend(ants_y)
y.extend(missing_y)
y.extend(pesti_y)
y.extend(healthy_y)

# Check the total number of labels
len(y)

# Concatenate merged features from each category
merged_features = np.concatenate((merged_ants_features, merged_missing_features, merged_pesti_features, merged_health_features), axis=0)

# Inspect merged features shape and data type
merged_features.shape
merged_features.dtype
merged_features

# Create a DataFrame with merged features
df = pd.DataFrame(merged_features)

# Preview the DataFrame
df.head()

# Add the labels as a new column in the DataFrame
df['label'] = y

# Standardize features
sc = StandardScaler()
X = sc.fit_transform(X)

# One-Hot Encode labels
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

# Split data into training and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=42)

# Further split the test set into validation and test sets
X_val,X_test,y_val,y_test = train_test_split(X_test,y_test,test_size = 0.5)

# Check the length of the validation set
len(X_val)

# Separate image and audio features for each dataset
X_image_train = X_train[:,:4096]
X_image_val = X_val[:,:4096]
X_image_test = X_test[:,:4096]

X_audio_train = X_train[:, -4096:]
X_audio_val = X_val[:, -4096:]
X_audio_test = X_test[:, -4096:]

# Define and compile the model for combined features (8192 input dimensions)
model = Sequential()
model.add(Dense(16, input_dim=8192, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on combined features
history = model.fit(X_train, y_train,validation_data = (X_val,y_val), epochs=20, batch_size=64)

# Define and compile the model for image features only (4096 input dimensions)
model = Sequential()
model.add(Dense(16, input_dim=4096, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on image features
history = model.fit(X_image_train, y_train,validation_data = (X_image_val,y_val), epochs=20, batch_size=64)
model.save('image_features_model.h5') 

### Image Predictions and Accuracy Calculation

# Generate predictions on the test set
y_image_pred = model.predict(X_image_test)

# Converting predictions to labels
image_pred = list()
for i in range(len(y_image_pred)):
    image_pred.append(np.argmax(y_image_pred[i]))

# Converting one-hot encoded test labels to labels
image_test = list()
for i in range(len(y_test)):
    image_test.append(np.argmax(y_test[i]))

# Accuracy Calculation
a = accuracy_score(image_pred,image_test)
print('Accuracy is:', a*100)

# Accuracy Plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

### Display Predictions and Classification Report
# Print predicted and true labels
print(image_pred)
print(image_test)

# Classification report
classes = ['healthy', 'ant problems', 'missing queen','pesticide']
print(classification_report(image_test, image_pred,  target_names=classes, digits=4))

### Confusion Matrix

# Confusion matrix visualization
cm = confusion_matrix(y_true = image_test,y_pred = image_pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.show()

### Audio Model Training

# Train the model on audio features
history = model.fit(X_audio_train, y_train,validation_data = (X_audio_val,y_val), epochs=20, batch_size=64)
model.save('audio_features_model.h5')

### Audio Predictions and Label Conversion

# Predict on the audio test set
y_audio_pred = model.predict(X_audio_test)

# Converting predictions to labels
audio_pred = list()
for i in range(len(y_audio_pred)):
   audio_pred.append(np.argmax(y_audio_pred[i]))

# Converting one-hot encoded test labels to labels
audio_test = list()
for i in range(len(y_test)):
   audio_test.append(np.argmax(y_test[i]))

### Audio Model Accuracy Calculation

# Calculate accuracy for audio predictions
a = accuracy_score(audio_pred,audio_test)
print('Accuracy is:', a*100)

### Audio Model Accuracy Plot

# Plot model accuracy over epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

### Display Audio Predictions and Classification Report

# Display predictions and test labels
print(audio_pred)
print(audio_test)

# Classification report for audio predictions
classes = ['healthy', 'ant problems', 'missing queen','pesticide']
print(classification_report(audio_test, audio_pred,  target_names=classes, digits=4))

### Confusion Matrix for Audio Predictions

# Confusion matrix visualization
print(classification_report(audio_pred, audio_test,  target_names=classes, digits=4))
cm = confusion_matrix(y_true = audio_test,y_pred = audio_pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Extract labels from DataFrame
output=df['label'].values
output

### Visual & Audio Predictions and Label Conversion

# Define Cross-Attention MultiModal Model
def BeeHealthModel(image_input_shape, audio_input_shape):
    dk = 64  # Dimension for query and key
    dv = 64  # Dimension for value

    # Define image and audio inputs
    image_input = tf.keras.Input(shape=image_input_shape, name="image_input")
    audio_input = tf.keras.Input(shape=audio_input_shape, name="audio_input")

    # Feature extraction with VGG16 for both image and audio inputs
    vgg_image = VGG16(weights="imagenet", include_top=False, pooling="avg")
    vgg_audio = VGG16(weights="imagenet", include_top=False, pooling="avg")

    feature_image = vgg_image(image_input)
    feature_audio = vgg_audio(audio_input)

    # Attention mechanism
    query = Dense(dk)(feature_image)         # Query from image features
    key = Dense(dk)(feature_audio)           # Key from audio features
    value = Dense(dv)(feature_audio)         # Value from audio features

    # Compute attention scores
    attention_scores = tf.keras.layers.Softmax()(tf.keras.layers.Dot(axes=-1)([query, key]) / tf.sqrt(tf.cast(dk, tf.float32)))

    # Compute context vector
    context_vector = tf.keras.layers.Dot(axes=1)([attention_scores, value])

    # Concatenate the image features and context vector
    feature_concatenated = Concatenate()([feature_image, context_vector])

    # Fully connected layers
    fc_layer1 = Dense(32, activation="relu")(feature_concatenated)
    fc_layer1_dropout = Dropout(0.5)(fc_layer1)
    fc_layer2 = Dense(16, activation="relu")(fc_layer1_dropout)
    fc_layer2_dropout = Dropout(0.5)(fc_layer2)

    # Output layer with softmax activation
    output = Dense(4, activation="softmax")(fc_layer2_dropout)

    # Define the model
    model = Model(inputs=[image_input, audio_input], outputs=output)
    return model

# Define input shapes (example shapes, adjust as needed)
image_input_shape = (224, 224, 3)
audio_input_shape = (224, 224, 3)

# Instantiate the cross-attention model
model = BeeHealthModel(image_input_shape, audio_input_shape)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Cross-Attention model
history = model.fit([X_image_train, X_audio_train], y_train, 
                    validation_data=([X_image_val, X_audio_val], y_val), 
                    epochs=20, batch_size=64)
model.save('cross_attention_bee_health_model.h5')

# Predict on the test set
y_pred = model.predict([X_image_test, X_audio_test])

# Convert predictions to labels
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))

# Convert one-hot encoded test labels to labels
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

### Accuracy Calculation

# Calculate and print accuracy
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)

### Model Accuracy Plot

# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

### Classification Report

# Display classification report
classes = ['healthy', 'ant problems', 'missing queen','pesticide']
print(classification_report(pred, test, target_names=classes, digits=4))

### Confusion Matrix

# Generate and display confusion matrix
cm = confusion_matrix(y_true = test,y_pred = pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.show()