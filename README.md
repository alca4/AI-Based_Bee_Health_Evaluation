# AI-Based Bee Health Evaluation

This repository is the implementation of [Developing an AI-based Integrated System for Bee Health Evaluation](https://arxiv.org/abs/2401.09988) by Andrew Liang.


Honey bees pollinate about one-third of the world's food supply, but bee colonies have alarmingly declined by nearly 40% over the past decade due to several factors, including pesticides and pests. 

Traditional methods for monitoring beehives, such as human inspection, are subjective, disruptive, and time-consuming. To overcome these limitations, artificial intelligence has been used to assess beehive health. However, previous studies have lacked an end-to-end solution and primarily relied on data from a single source, either bee images or sounds. 

This study introduces a comprehensive system consisting of bee object detection and health evaluation. Additionally, it utilizes a combination of visual and audio signals to analyze bee behaviors. An Attention-based Multimodal Neural Network (AMNN) is developed to adaptively focus on key features from each type of signal for accurate bee health assessment. By seamlessly integrating AMNN with image and sound data in a comprehensive bee health monitoring system, this approach provides a more efficient and non-invasive solution for the early detection of bee diseases and the preservation of bee colonies.

This end-to-end solution includes 

1. Data acquisition and annotation for model development 
2. Bee object detection to detect bee presence in images and audio clips
3. Bee health assessment based on either images or audio clips
4. An attention-based multimodal neural network (AMNN) that combines both visual and audio signals to assess the health of beehives

The system is the first to integrate bee health assessment with bee object detection using paired image and audio data.

## Bee Images Object Detection

A YOLOv5 model was developed for the localization and cropping of bees in images.

The code can be found in Bee_Image_Object_Detection.ipynb.

## Bee Audio Object Detection

Four 1D CNN models were developed to identify bees in audio clips. Each model used a distinct audio feature: Mel Spectrogram, MFCC, STFT, or Chromagram. 

The code can be found in Bee_Audio_Object_Detection.ipynb.

## Bee Images Health Assessment

The visual bee health evaluation focused on classifying cropped bee images into various health categories. In this step, four distinct deep learning models were implemented as outlined, including CNN, Inceptionv3, MobileNetv2 and VGG16.

The code can be found in Bee_Image_Health_Assessment_CNN.ipynb and Bee_Image_Health_Assessment_Inception_MobileNet.ipynb.

## Bee Audio Health Assessment

1D CNN, 2D CNN, LSTM, and VGG16 models were also developed for bee health classification through audio analysis.

The code can be found in Bee_Audio_Health_Assessment_1D_CNN.ipynb and Bee_Audio_Health_Assessment_2D_CNN_LSTM.ipynb.

## Attention-based Multimodal Neural Network (AMNN)

Previous models for bee health assessment focused solely on either bee images or audio. To address the limitations of these isolated approaches, the AMNN was proposed to merge both bee visual and auditory information. By incorporating an attention mechanism, the model could dynamically focus on crucial features in each modality. This adaptability enabled a comprehensive understanding of bee behavior and improved bee health assessment accuracy.

The code can be found in Bee_Health_VGG16_AMNN.ipynb, including VGG16 for individual visual and audio modality, and combined signals.
