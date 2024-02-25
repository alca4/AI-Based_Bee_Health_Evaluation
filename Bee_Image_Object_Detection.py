# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/ultralytics/yolov5
# %cd yolov5

# Commented out IPython magic to ensure Python compatibility.
# %pip install -qr requirements.txt

# Commented out IPython magic to ensure Python compatibility.

from yolov5 import utils #go to folder Bee_Object_detection
import torch

import utils
# import core
from IPython import display
from IPython.display import clear_output
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


# %matplotlib inline
display = utils.notebook_init()

!cat data/Bee.yaml
print ('---------------------------------------------')

!python train.py --batch 64 --epochs 50 --data 'data/Bee.yaml' --weights 'yolov5m6.pt' --project 'run_bee' --name 'feature_extraction' --cache --freeze 12
# clear_output()

display.Image(f"run_bee/feature_extraction9/results.png")

!python detect.py --weights 'run_bee/feature_extraction7/weights/best.pt'  --conf 0.6 --source './Downloads/Bee_Object_Detection/yolov5/test/images' --project 'run_bee' --name 'detect_test' --augment --line=3 --save-txt --save-crop