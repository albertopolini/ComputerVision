# ComputerVision

## Overview
This repository contains implementations of main deep learning computer vision architectures using PyTorch. The goal is to provide a comprehensive resource for understanding and experimenting with the main models in computer vision.

## Table of Contents

1. [Introduction](#introduction)
2. [Architectures](#architectures)
3. [Docker](#docker)
4. [License](#license)

## Introduction

Deep learning has revolutionized computer vision, and this repository aims to gather implementations of popular architectures, making it easy for researchers, practitioners, and enthusiasts to explore, learn, and experiment with these models.

## Architectures

The following architectures are currently implemented:

- [x] **ResNet**: Deep Residual Networks
- [x] **VGG**: Visual Geometry Group Networks
- [x] **AlexNet**: AlexNet: ImageNet Classification with Deep Convolutional Neural Networks
- [ ] **DenseNet**: Densely Connected Convolutional Networks
- [ ] **Inception**: Going Deeper with Convolutions
- [ ] **MobileNet**: Efficient Convolutional Neural Networks for Mobile Vision Applications



## Docker


1. cd into the docker folder
2. Run the command

```
docker build -t <image_name> --build-arg token_name=<token> . 
```
    
To build the container run:
    
```
docker run -it \
--gpus all \
--name <instance_name> \
-p 8888:8888 \
-v <Path/to/notebook/folder>:/home/Notebooks ^\
-v <Path/to/data/folder>:/home/Data \
-v <Path/to/outputs/folder>:/home/Outputs \
-v <Path/to/scripts/folder>:/home/Scripts \
<image_name>
```

If you are using Rancher Desktop you should map the folder paths with 'mnt/c' 

## License
This project is licensed under the MIT License - see the (LICENSE)[#LICENSE] file for details.

