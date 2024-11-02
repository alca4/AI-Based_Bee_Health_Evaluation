import os
import random
import shutil
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import librosa
from keras.models import load_model
from PIL import Image
# import jsonp

# Define the base directory for all file paths
base_dir = './'

def showVideo(file_name):
    # Set up the specific paths based on the base directory
    file_name = "bee.mp4"
    video_path = os.path.join(base_dir, "static/uploads/"+file_name)
    frames_folder = os.path.join(base_dir, 'frames')
    cropped_folder = os.path.join(base_dir, 'cropped')
    model_path = os.path.join(base_dir, 'bee_health_multimodel.h5')
    print(f'video_path={video_path} ')
    extract_frames_and_audio(video_path, frames_folder)
    # Ensure necessary folders exist
    if not os.path.exists(frames_folder):
        print(f"The folder {frames_folder} does not exist. Please create it first.")
    if not os.path.exists(cropped_folder):
        os.makedirs(cropped_folder, exist_ok=True)
    image_path, audio_path = pick_image_and_audio(frames_folder, cropped_folder)
    if image_path and audio_path:
        return predict_health_condition(image_path, audio_path, model_path)
    return []



# Function to extract frames and audio
def extract_frames_and_audio(video_path, output_folder, interval=10):
    if not os.path.exists(output_folder):
        print(f"Output folder '{output_folder}' does not exist.")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    video = VideoFileClip(video_path)
    video_duration = video.duration
    frame_count, extracted_count = 0, 0
    success, frame = cap.read()

    while success:
        start_time = frame_count / fps
        if start_time + interval > video_duration:
            print("Reached end of video.")
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{extracted_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame: {frame_filename}")

            end_time = start_time + interval
            audio_clip = video.subclip(start_time, end_time)
            audio_filename = os.path.join(output_folder, f"audio_{extracted_count}.wav")
            audio_clip.audio.write_audiofile(audio_filename, codec='pcm_s16le')
            print(f"Saved audio clip: {audio_filename}")
            extracted_count += 1

        success, frame = cap.read()
        frame_count += 1

    cap.release()
    video.close()

# Function to randomly pick and copy audio to cropped folder
def pick_image_and_audio(folder_path, cropped_folder):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    if not image_files or not audio_files:
        print("No image or audio files found.")
        return None, None

    selected_image = random.choice(image_files)
    selected_audio = random.choice(audio_files)
    image_path = os.path.join(folder_path, selected_image)
    audio_path = os.path.join(folder_path, selected_audio)
    cropped_audio_path = os.path.join(cropped_folder, selected_audio)
    shutil.copy(audio_path, cropped_audio_path)
    print(f"Audio file saved to: {cropped_audio_path}")
    return image_path, audio_path

# Function to preprocess audio for prediction using STFT
def extract_stft(audio_file, sr=22000, n_fft=1024, hop_length=256):
    audio, _ = librosa.load(audio_file, sr=sr)
    stft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    return np.resize(stft_db, (4096,)).reshape(1, 4096)

# Function to preprocess image for prediction
def preprocess_image(image_file, target_size=(64, 64)):
    img = Image.open(image_file).convert('L')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array.flatten().reshape(1, 4096)

# Load the model and make predictions
def predict_health_condition(image_path, audio_path, model_path):
    model = load_model(model_path)
    audio_features = extract_stft(audio_path)
    image_features = preprocess_image(image_path)
    prediction = model.predict([image_features, audio_features])
    classes = ['healthy', 'ant problems', 'missing queen', 'pesticide']
    values = []
    response = [classes,values]
    for i, label in enumerate(classes):
        print(f"{label}: {prediction[0][i]*100:.2f}%")
        values.append(round(prediction[0][i]*100, 2))
    print(f"response: {response}")
    return [classes,values]
    # prediction[0][i]*100:.2f
    # for i, label in enumerate(classes):
    #     print(f"{label}: {prediction[0][i]*100:.2f}%")
    print(f"Predicted bee health condition: {classes[np.argmax(prediction)]}")

# Usage

