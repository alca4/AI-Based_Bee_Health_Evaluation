import librosa
import numpy as np
from keras.models import load_model
import os

# Load your trained model
model = load_model('bee_health_multimodel.h5')

# Path to the audio file
# audio_file = 'test.wav'


base_dir = './'

def showAudio(file_name):
    # Set up the specific paths based on the base directory
    # file_name = "bee.mp4"
    audio_file = os.path.join(base_dir, "static/uploads/"+file_name)
    frames_folder = os.path.join(base_dir, 'frames')
    cropped_folder = os.path.join(base_dir, 'cropped')
    model_path = os.path.join(base_dir, 'bee_health_multimodel.h5')
    print(f'audio_file={audio_file} ')

    # Extract Mel spectrogram from the audio file
    mel_spectrogram = extract_mel_spectrogram(audio_file)

    # Resize and flatten the Mel spectrogram to match the model's expected input size
    # Assuming the model expects a 4096-dimensional input
    mel_spectrogram_resized = np.resize(mel_spectrogram, (4096,))
    mel_spectrogram_resized = mel_spectrogram_resized.reshape(1, 4096)

    # Prepare a dummy image input since the model expects two inputs
    dummy_image_input = np.zeros((1, 4096))  # Match the shape of the image input (1, 4096)

    # Make the prediction by passing both the dummy image input and the actual audio input
    prediction = model.predict([dummy_image_input, mel_spectrogram_resized])

    # Define the class labels
    classes = ['healthy', 'ant problems', 'missing queen', 'pesticide']
    values = []
  
    # Output the probability for each class label
    for i, label in enumerate(classes):
        print(f"{label}: {prediction[0][i]*100:.2f}%")
        values.append(round(prediction[0][i]*100, 2))

    response = [classes,values]
    # Output the predicted class
    # predicted_class = np.argmax(prediction)
    # print(f"Predicted bee health condition: {classes[predicted_class]}")
    return response

# Function to extract features (e.g., Mel spectrogram) from the audio file
def extract_mel_spectrogram(audio_file, sr=22050, n_mels=128, hop_length=512):
    # Load the audio file
    audio, sample_rate = librosa.load(audio_file, sr=sr)

    # Generate Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, hop_length=hop_length)

    # Convert the Mel spectrogram to decibel units
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_spectrogram_db
