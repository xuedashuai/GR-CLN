# GR-CLN: a Generative-rehearsal Continual Learning Network for Trajectory Prediction
Our paper is going to be published 

## Setup：
The code was written in the following environment:  
- python 3.7.11  
- pytorch 1.10.0  
- cuda 11.3  
- cudnn 8.2.0  

## Preparation for data:
The raw data of INTERACTION is downloadable at https://interaction-dataset.com/
- Our data interface is improved based on D-GSM (https://github.com/BIT-Jack/D-GSM). 
- Put the raw data into `./raw`
- Run `/datasets/preprocess_for_learning.py` to preprocess the data for GR-CLN.
- Run `/datasets/gen_temp.py` to enhance the efficiency of data loading during multiple training sessions in continual learning. 

## Using the code:
To use the pretrained models at `./trained_models` and evaluate the models performance run:  
```
python evaluate.py
```
