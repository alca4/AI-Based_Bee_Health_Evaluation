import librosa
import numpy
from keras.models import load_model
import os
model = load_model("bee_health_multimodel.h5")
print(f'model = {model}')
