import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import cv2
import os
from moviepy.editor import VideoFileClip

def extract_frames_and_audio(video_path, output_folder, interval=10):
    # Check if the output directory exists, if not, exit the function
    if not os.path.exists(output_folder):
        print(f"Output folder '{output_folder}' does not exist. Please create the folder first.")
        return

    # Open the video file for frame extraction
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)  # Calculate frame interval in frames

    # Load video with moviepy for audio extraction
    video = VideoFileClip(video_path)
    video_duration = video.duration

    frame_count = 0
    success, frame = cap.read()
    extracted_count = 0  # To keep track of the number of extractions

    while success:
        # Check if there is enough remaining duration for a 10-second clip
        start_time = (frame_count / fps)
        if start_time + interval > video_duration:
            print("Reached end of video. Stopping extraction.")
            break

        # Process frames and audio clips at the specified interval
        if frame_count % frame_interval == 0:
            # Save the frame as an image file
            frame_filename = os.path.join(output_folder, f"frame_{extracted_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame: {frame_filename}")

            # Extract the corresponding 10-second audio clip
            end_time = start_time + 10
            audio_clip = video.subclip(start_time, end_time)
            audio_filename = os.path.join(output_folder, f"audio_{extracted_count}.wav")
            audio_clip.audio.write_audiofile(audio_filename, codec='pcm_s16le')
            print(f"Saved audio clip: {audio_filename}")

            extracted_count += 1  # Increment the extracted count

        # Read next frame and increment frame count
        success, frame = cap.read()
        frame_count += 1

    # Release resources
    cap.release()
    video.close()
    print("Frame and audio extraction complete.")

# Usage
video_path = 'bee.mp4'
output_folder = 'frames'
extract_frames_and_audio(video_path, output_folder, interval=10)

import os
import random
import shutil

def pick_image_and_audio(folder_path, cropped_folder):
    # Get all image and audio files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    # Check if there are images and audio files
    if not image_files or not audio_files:
        print("No images or audio files found in the folder.")
        return None, None


    selected_image = random.choice(image_files)
    selected_audio = random.choice(audio_files)

    # Construct the full paths
    image_path = os.path.join(folder_path, selected_image)
    audio_path = os.path.join(folder_path, selected_audio)

    # Output the file paths
    print(f"Selected image path: {image_path}")
    print(f"Selected audio path: {audio_path}")

    # Copy the audio file to the cropped folder
    cropped_audio_path = os.path.join(cropped_folder, selected_audio)
    shutil.copy(audio_path, cropped_audio_path)
    print(f"Audio file saved to: {cropped_audio_path}")

    return image_path, audio_path

# Usage
folder_path = 'frames'
cropped_folder = 'cropped'

# Ensure the cropped folder exists
os.makedirs(cropped_folder, exist_ok=True)

# Pick and save the audio file
image_path, audio_path = pick_image_and_audio(folder_path, cropped_folder)

import os
import random
import subprocess

def pick_image_and_audio(folder_path):
    # Get all image and audio files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    # Check if there are images and audio files
    if not image_files or not audio_files:
        print("No images or audio files found in the folder.")
        return None, None


    selected_image = random.choice(image_files)
    selected_audio = random.choice(audio_files)

    # Construct the full paths
    image_path = os.path.join(folder_path, selected_image)
    audio_path = os.path.join(folder_path, selected_audio)

    # Output the file paths
    print(f"Selected image path: {image_path}")
    print(f"Selected audio path: {audio_path}")

    return image_path, audio_path

# Usage
folder_path = '/root/Audio_Bee_NoBee_out/frames'
image_path, audio_path = pick_image_and_audio(folder_path)

# Run the command with the selected image path
command = [
    'python', '/root/Bee_Object_Detection/yolov5/detect.py',
    '--weights', '/root/Bee_Object_Detection/yolov5/run_bee/feature_extraction7/weights/best.pt',
    '--conf', '0.6',
    '--source', image_path,
    '--project', '/root/Bee_Object_Detection/yolov5/run_bee',
    '--name', 'detect_bee',
    '--augment',
    '--line=3',
    '--save-txt',
    '--save-crop'
]

# Execute the command
subprocess.run(command)

# Path to the cropped images
cropped_images_path = '/root/Bee_Object_Detection/yolov5/run_bee/detect_bee/crops/bee'

# Check if the directory exists and contains images
if os.path.isdir(cropped_images_path):
    cropped_images = [f for f in os.listdir(cropped_images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if cropped_images:

        cropped_image = random.choice(cropped_images)
        cropped_image_path = os.path.join(cropped_images_path, cropped_image)
        print(f"cropped image path: {cropped_image_path}")
    else:
        print("No cropped images found in the 'crops/bee' directory.")
else:
    print("The 'crops/bee' directory does not exist.")

import shutil

# Define the destination folder
destination_folder = '/root/Audio_Bee_NoBee_out/cropped'

# Check if the selected cropped image exists
if cropped_image:
    # Construct the new file name by adding 'cropped' before the file extension
    filename, file_extension = os.path.splitext(cropped_image)
    new_filename = f"{filename}_cropped{file_extension}"

    # Define the destination path with the new file name
    destination_path = os.path.join(destination_folder, new_filename)

    # Copy the file to the destination with the new name
    shutil.copy(cropped_image_path, destination_path)
    print(f"Saved cropped image as: {destination_path}")
else:
    print("No cropped image to save.")

import os
import random
import librosa
import numpy as np
from keras.models import load_model
from PIL import Image

# Define paths to the folders containing images and audio
folder_path = '/root/Audio_Bee_NoBee_out/cropped'
model_path = '/root/bee_health_multimodel.h5'

# Load your trained model
model = load_model(model_path)

# Function to randomly select an audio and image file
def selected_files(folder_path):
    # Get all image and audio files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    if not image_files or not audio_files:
        print("No image or audio files found.")
        return None, None

    # Pick a random image and audio file
    selected_image = random.choice(image_files)
    selected_audio = random.choice(audio_files)
    return os.path.join(folder_path, selected_image), os.path.join(folder_path, selected_audio)

# Feature extraction functions for audio and image
def extract_mel_spectrogram(audio_file, sr=22050, n_mels=256, hop_length=512):
    audio, sample_rate = librosa.load(audio_file, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, hop_length=hop_length)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return np.resize(mel_spectrogram_db, (4096,)).reshape(1, 4096)

def preprocess_image(image_file, target_size=(64, 64)):
    img = Image.open(image_file).convert('L')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array.flatten().reshape(1, 4096)


image_path, audio_path = selected_files(folder_path)
if image_path and audio_path:

    # Extract features and predict
    audio_features = extract_mel_spectrogram(audio_path)
    image_features = preprocess_image(image_path)

    # Make prediction
    prediction = model.predict([image_features, audio_features])

    # Class labels
    classes = ['healthy', 'ant problems', 'missing queen', 'pesticide']

    # Print probabilities for each class
    for i, label in enumerate(classes):
        print(f"{label}: {prediction[0][i]*100:.2f}%")

    # Print the predicted class
    predicted_class = np.argmax(prediction)
    print(f"Predicted bee health condition: {classes[predicted_class]}")