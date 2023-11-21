# Airbus_ship_detection

## Overview
This repo includes data analisis and code to building a model for Kaggle Airbus Ship Detection Challenge
link: https://www.kaggle.com/competitions/airbus-ship-detection/overview

Main files:
 * `model.py` - the model
 * `train.py` - trains the model
 * `test.py` - trains the model(s)
 * `requirements.txt` - required python modules
 * `airbus-ship-detection.ipynb` - exploratory data analysis

## Description of solution
### Data exploration
In this competition, we needed to locate ships in images. More than half images do not contain ships, and those that do may contain multiple ships. 
File train_ship_segmentations_v2.csv contains id and encoded pixels with places of ships. So we need to decode these pixels into mask the same size as our images.

Example of data:
![image](https://github.com/pulsatil1/Airbus-Ship-Detection/assets/70263951/21d9134d-4aa0-4866-a82b-b6c2df8e7594)


We can reduce the images size to facilitate the learning process for the neural network. But the ships on images might be very small, so we can reduce images size only a little.
Also, because of the dataset isn't balanced, we created a balanced train and validation datasets.

![image](https://github.com/pulsatil1/Airbus-Ship-Detection/assets/70263951/a7d77d7b-540c-48a5-9a48-5f732d18b44b)


### Model
We used a model with U-Net architecture that is a good choice for the segmentation task.
We also used Dropout layers and data augmentation, to avoid overfitting.
Trained the model using GPU.

![image](https://github.com/pulsatil1/Airbus-Ship-Detection/assets/70263951/c5c52f4e-6e33-4aff-82d3-8d5bc5bbc8a5)


### Loss function and metric
We used the Dice coefficient as a main metric. But we can't use it as a loss function, because the Dice coefficient increases when the model works better, and the loss function needs to decrease when the model shows better results. So I chose Focal Loss as a loss function.

### Results of predicting 

![image](https://github.com/pulsatil1/Airbus-Ship-Detection/assets/70263951/84bd3f72-ce98-4397-965e-b274539c3eeb)
![image](https://github.com/pulsatil1/Airbus-Ship-Detection/assets/70263951/74e9ddfe-5a4d-43ca-b123-021412dad44d)
![image](https://github.com/pulsatil1/Airbus-Ship-Detection/assets/70263951/94ca484c-5689-48e8-ad69-fc2374c301a8)
![image](https://github.com/pulsatil1/Airbus-Ship-Detection/assets/70263951/9cb6f960-3417-4fef-bcd7-4480d7913843)

