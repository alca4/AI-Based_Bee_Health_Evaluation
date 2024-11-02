# AI-Based Bee Health Evaluation

This repository is the implementation of [Developing a multimodal system for bee object detection and health assessment](https://doi.org/10.1109/ACCESS.2024.3464559) by Andrew Liang.

Honey bees pollinate about one-third of the world's food supply, but bee colonies have alarmingly declined by nearly 40% over the past decade due to several factors, including pesticides and pests. 

Traditional methods for monitoring beehives, such as human inspection, are subjective, disruptive, and time-consuming. To overcome these limitations, artificial intelligence has been used to assess beehive health. However, previous studies have lacked an end-to-end solution and primarily relied on data from a single source, either bee images or sounds. 

This study introduces a comprehensive system consisting of bee object detection and health evaluation. Additionally, it utilizes a combination of visual and audio signals to analyze bee behaviors. A Cross-attention-based Multimodal Neural Network (CAMNN) is developed to adaptively focus on key features from each type of signal for accurate bee health assessment. By seamlessly integrating CAMNN with image and sound data in a comprehensive bee health monitoring system, this approach provides a more efficient and non-invasive solution for the early detection of bee diseases and the preservation of bee colonies. A live streaming system and mobile application allow beekeepers to observe their hive and receive health assessments remotely, facilitating early intervention when hive stressors appear.

This end-to-end solution includes 

1. Data acquisition and annotation for model development 
2. Bee object detection to detect bee presence in images and audio clips
3. A cross-attention-based multimodal neural network (CAMNN) that combines both visual and audio signals to assess the health of beehives
4. A near-real-time live streaming system to send videos of hive activities to beekeepers remotely.
5. A mobile application where users can upload videos to receive health assessment reports, access real-time weather updates, and review a history of past health assessments.

## Bee Images Object Detection

A YOLOv5 model was developed for the localization of bees in images.

The code can be found in Models/Bee_Image_Object_Detection.py

## Bee Audio Object Detection

Four 1D CNN models were developed to identify bees in audio clips. Each model used a distinct audio feature: Mel Spectrogram, MFCC, STFT, or Chromagram. 

The code can be found in Models/Bee_Audio_Object_Detection.py

## Cross-Attention-based Multimodal Neural Network (CAMNN)

CAMNN was proposed to merge both bee visual and auditory information. By incorporating an attention mechanism, the model could dynamically focus on crucial features in each modality. This adaptability enabled a comprehensive understanding of bee behavior and improved bee health assessment accuracy.

The code can be found in Models/Bee_Health_Assessment.py

## Live Streaming

The streaming was performed with an Arducam autofocus camera and a PoP Voice Profesional noise-reducing microphone. The video is captured using libcamera-vid, and the audio is captured using arecord. The code contains two loops: one generates video and audio files of length 10s, and the other combines them according to the HTTP Live Streaming (HLS) protocol. .m3u8 files are combined into a large playlist file. 

The code can be found in Live Streaming/live-streaming.sh

## Mobile Application

The Android app was built using React Native. The OpenWeatherMap API provides live weather updates. The uploaded data are run through the trained models to generate hive health status using Python and Flask. 

The code can be found under the Mobile Application folder
