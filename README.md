# Bee-Project


Bees pollinate over 80% of plants, but bee colonies have been experiencing a devastating 39.7% annual loss over the past 11 years. 

To address this pressing issue, I collected the paired data from three apiaries in California and developed an innovative end-to-end deep learning-based system. This end-to-end solution includes 

1. Data acquisition and annotation for model development 
2. Bee object detection to detect bee presence in images and audio clips
3. Bee health assessment to assess bee health in images and audio clips
4. Multi-modal Visual-Audio Deep Neural Network to combine both visual and audio signals to assess beehives’ health
5. Web app deployment with near real-time streaming and bee health classification

The system is the first to integrate bee health assessment with bee object detection using paired image and audio data in research.

## Bee images object detection

A YOLOv5 model19 was developed for the localization and cropping of bees in images.

Code can be found in Bee_Image_Object_Detection.ipynb.

## Bee audio object detection

Four 1D CNN models were developed to identify bees in audio clips. Each model used a distinct audio feature:
Mel Spectrogram, MFCC, STFT, or Chromagram. 

Code can be found in Bee_Audio_Object_Detection.ipynb.

## Bee images health assessment

The visual bee health evaluation focused on classifying cropped bee images into various health categories. In this
step, four distinct deep learning models were implemented as outlined, including CNN, Inceptionv3, MobileNetv2 and VGG16.

Code can be found in Bee_Image_Health_Assessment_CNN.ipynb and Bee_Image_Health_Assessment_Inception_MobileNet.ipynb.

## Bee audio health assessment

1D CNN, 2D CNN, LSTM, and VGG16 models were also developed for bee health classification through audio analysis.

Code can be found in Bee_Audio_Health_Assessment_1D_CNN.ipynb and Bee_Audio_Health_Assessment_2D_CNN_LSTM.ipynb.


