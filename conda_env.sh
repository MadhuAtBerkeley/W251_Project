#!/bin/bash -l
conda create -y -n w251_project python=3.9 jupyter
conda activate w251_project
conda install dlib
conda install -c akode face_recognition_models
conda install -c conda-forge face_recognition
conda install certifi click dotmap tqdm Pillow numpy imutils requests
conda install -c conda-forge pytorch  torchvision


