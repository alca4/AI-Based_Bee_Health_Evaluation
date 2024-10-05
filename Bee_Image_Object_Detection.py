# Clone YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5

# Navigate to the YOLOv5 directory
%cd yolov5

# Install necessary dependencies
!pip install -r requirements.txt

from yolov5 import utils #go to folder Bee_Object_detection
import torch

import utils
from IPython import display
from IPython.display import clear_output
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# Initialize utilities for better notebook interaction
display = utils.notebook_init()

# View the Bee dataset configuration file
!cat data/Bee.yaml
print ('---------------------------------------------')

# Train YOLOv5 model
# Using `yolov5m6.pt` as a starting point with 50 epochs and freezing the first 12 layers
!python train.py --batch 64 --epochs 50 --data 'data/Bee.yaml' --weights 'yolov5m6.pt' --project 'run_bee' --name 'feature_extraction' --cache --freeze 12

# Clear output
clear_output()

# Display training results
display.Image(f"run_bee/feature_extraction9/results.png")

# Test detection with the trained model
# Change 'run_bee/feature_extraction7/weights/best.pt' to the correct path to the best model from the training phase
!python detect.py --weights 'run_bee/feature_extraction7/weights/best.pt'  --conf 0.6 --source './Downloads/Bee_Object_Detection/yolov5/test/images' --project 'run_bee' --name 'detect_test' --augment --line=3 --save-txt --save-crop

# View detection results
display.Image(f"run_bee/detect_test/exp/image0.jpg")
