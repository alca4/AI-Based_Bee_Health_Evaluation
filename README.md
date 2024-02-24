# Bee-Project

This repository is the implementation of [Developing an AI-based Integrated System for Bee Health Evaluation] (https://arxiv.org/abs/2401.09988)
Andrew Liang

Bees pollinate over 80% of plants, but bee colonies have been experiencing a devastating 39.7% annual loss over the past 11 years. 

To address this pressing issue, I collected the paired data from three apiaries in California and developed an innovative end-to-end deep learning-based system. This end-to-end solution includes 

1. Data acquisition and annotation for model development 
2. Bee object detection to detect bee presence in images and audio clips
3. Bee health assessment to assess bee health in images and audio clips
4. Multi-modal Visual-Audio Deep Neural Network to combine both visual and audio signals to assess beehives’ health
5. Web app deployment with near real-time streaming and bee health classification

The system is the first to integrate bee health assessment with bee object detection using paired image and audio data in research.

## Bee Images Object Detection

A YOLOv5 model19 was developed for the localization and cropping of bees in images.

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

The code can be found in AMNN.ipynb, including VGG16 for individual visual and audio modality, and combined signals.
