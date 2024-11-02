from keras.models import load_model
from PIL import Image
import numpy as np

# Load your trained model
model = load_model('bee_health_multimodel.h5')

# Path to the image file
image_file = 'image.png'

# Function to preprocess the image
def preprocess_image(image_file, target_size=(64, 64)):
    # Load the image and convert it to grayscale (1 channel) to match the model input
    img = Image.open(image_file).convert('L')  # Convert to grayscale (L mode is 1 channel)

    # Resize the image to the target size (assume 64x64)
    img = img.resize(target_size)

    # Convert the image to a numpy array and flatten it
    img_array = np.array(img)

    # Normalize the image (values between 0 and 1)
    img_array = img_array / 255.0

    # Flatten the image to match the expected input size (4096,)
    img_flattened = img_array.flatten().reshape(1, -1)  # Shape (1, 4096)

    return img_flattened

# Preprocess the image to get a 4096-dimensional input
image_input = preprocess_image(image_file)

# Prepare a dummy input since the model expects two inputs
dummy_input = np.zeros((1, 4096))  # Adjust the size if your model expects different input dimensions

# Make the prediction by passing both the image and the dummy input to the model
prediction = model.predict([image_input, dummy_input])

# Define the class labels
classes = ['healthy', 'ant problems', 'missing queen', 'pesticide']

# Output the probability for each class label
for i, label in enumerate(classes):
    print(f"{label}: {prediction[0][i]*100:.2f}%")

# Output the predicted class
predicted_class = np.argmax(prediction)
print(f"Predicted bee health condition: {classes[predicted_class]}")