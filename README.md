# GlassBoxAI-CNN

**Author:** Matthew Abbott (2025)

This repository provides CUDA-accelerated implementations of Convolutional Neural Networks (CNNs). It features two primary modules:
- **cnn.cu:** A full-featured, direct CUDA implementation for learning and inference.
- **facaded_cnn.cu:** A modernized facade for teaching, rapid prototyping, and research access, with friendlier C++ abstractions.

The project aims for **transparency, reproducibility, and extensibility** for deep neural networks on the GPU, providing researchers and advanced users fine control over the network and training process.

---

## Table of Contents

- [License](#license)
- [Features](#features)
- [Requirements](#requirements)
- [cnn.cu](#cnncu)
  - [Usage](#usage)
  - [Arguments](#arguments)
  - [Public Methods](#public-methods)
- [facaded_cnn.cu (CNN Facade)](#facaded_cnncu-cnn-facade)
  - [Usage](#usage-1)
  - [Arguments](#arguments-1)
  - [Public Methods](#public-methods-1)
- [Overview & Notes](#overview--notes)
- [License](#license-1)

---

## Features

- Multi-layer convolutional neural networks with multi-class (softmax/cross-entropy) and regression.
- CUDA support for forward and backward passes, including full backpropagation and Adam optimizer.
- Numerical stability for training (clipping, stabilizing softmax).
- Flexible architecture definition (layer counts, sizes, activations).
- Model save/load to disk for training persistence.
- Facade API for rapid research or extension with PyTorch/NumPy compatibility in mind.

---

## Requirements

- NVIDIA GPU with CUDA support (tested with CUDA 11+)
- CUDA toolkit (nvcc, CUDA libraries)
- C++11 or later
- Optional: CMake (for your own build system)

---

## cnn.cu

### High-Level Description

A "bare-metal" CUDA implementation of a convolutional neural network suitable for fast direct training or prediction.  It implements:

- Convolution, pooling, and fully connected layers
- Dropout, Adam optimizer, ReLU+derivative, and stable Softmax
- Model IO and introspection
- Utility command-line interface

### Usage

```bash
nvcc -o cnn_cuda cnn.cu -lcurand

# Create new model
./cnn_cuda create [options] --save=MODEL_FILE

# Train model
./cnn_cuda train --model=MODEL_FILE --image=IMAGE_FILE --target=V1,V2,... [options] --save=NEW_MODEL_FILE

# Predict
./cnn_cuda predict --model=MODEL_FILE --image=IMAGE_FILE [options]

# Show model info
./cnn_cuda info --model=MODEL_FILE

# Help
./cnn_cuda help
```

### Arguments

#### Global/CLI Operations

| Argument        | Description                                                             |
|-----------------|-------------------------------------------------------------------------|
| `create`        | Create a new CNN and save it to file.                                   |
| `train`         | Train on a given image-target tuple.                                    |
| `predict`       | Run the model and output predictions.                                   |
| `info`          | Show architecture/config details for existing model.                    |
| `help`          | Display all options (usage instructions).                               |
| `--save=file`   | File to save model after operation.                                     |
| `--model=file`  | File to load model from.                                                |
| `--image=file`  | Input data file (see model documentation for format).                   |
| `--target=v1,v2,...` | Target output(s) for training.                                     |
| Options         | Various arguments for network architecture (not shown exhaustively).     |

#### Model/Constructor Arguments

- `inputWidth`: Width of input image (int)
- `inputHeight`: Height of input image (int)
- `inputChannels`: Channels in input image (int)
- `convFilters`, `kernelSizes`, `poolSizes`: vectors; specify number/sizes of conv/pooling layers
- `fcSizes`: Fully connected layer structure (vector of ints)
- `outputSize`: Output classification dimension or regression targets (int)
- `learningRate`, `dropoutRate`: Training hyperparameters (double, optional)

### Public Methods (Class: `TConvolutionalNeuralNetworkCUDA`)

- `Predict(const double* imageData, double* result)`
  - Forward-passes the image data and fills result with output probabilities or values.
- `double TrainStep(const double* imageData, const double* target)`
  - Single SGD/Adam training step. Returns loss.
- `bool Save(const char* filename)`
  - Save the current network state to a file.
- `static TConvolutionalNeuralNetworkCUDA* Load(const char* filename)`
  - Load a network state from file.

**Introspection:**
- `GetInputWidth()`, `GetInputHeight()`, `GetInputChannels()`, `GetOutputSize()`
- `GetNumConvLayers()`, `GetNumFCLayers()`
- `GetLearningRate()`, `GetDropoutRate()`
- `GetConvFilters()`, `GetKernelSizes()`, `GetPoolSizes()`, `GetFCSizes()`

---

## facaded_cnn.cu (CNN Facade)

### High-Level Description

A research- and education-friendly C++/CUDA class, with a Python-esque API. Features more modular OOP, ease of access to features, layer statistics, and model/package serialization.

Ideal for building advanced diagnostics, custom research applications, or rapid iteration.

### Usage

```bash
nvcc -O2 -o cnn_facade_cuda facaded_cnn.cu
```
(As a library: see constructor docs below.)

### Arguments

#### Constructor Arguments (Class: `CNNFacade::TCNNFacade`)

- `InputWidth`: Image width
- `InputHeight`: Image height
- `InputChannels`: Number of channels (e.g., 3 = RGB)
- `ConvFilters`, `KernelSizes`, `PoolSizes`: Layer structure as vector<int>
- `FCLayerSizes`: fully connected sizes as vector<int>
- `OutputSize`: Output dimension
- `ALearningRate` (default 0.001): Learning rate
- `ADropoutRate` (default 0.25): Dropout probability

#### Training/Prediction Methods

- `Darray Predict(TImageData& Image)`
  - Predict probabilities/outputs for input image (see TImageData constructor).
- `double TrainStep(TImageData& Image, const Darray& Target)`
  - Run training iteration. Returns cross-entropy loss.
- `void SaveModel(const std::string& Filename)`
  - Save model weights/architecture to disk.
- `void LoadModel(const std::string& Filename)`
  - Load from serialized file.

#### Public/Introspection Methods

- `int GetNumLayers()`
- `int GetNumConvLayers()`
- `int GetNumFCLayers()`
- `TLayerConfig GetLayerConfig(int LayerIdx)`
- `void SetTrainingMode(bool)`
- `D2array GetFeatureMap(int LayerIdx, int FilterIdx)`
  - Get the output of a given filter in a given convolutional layer (after last forward pass).

#### Data Structures

- `CNNFacade::TImageData`
  - Holds image channel-major `(channels, height, width)` vector data.
- `CNNFacade::TLayerConfig`
  - Struct with all meta info about each layer (type, sizes, channels, etc).

---

## Overview & Notes

- **No icons or branding**—Just clean code and documentation.
- The code provides full control, as expected for research-grade or deep debugging environments.
- For input/output, see the code and comments for expected file formats.
- No external neural network frameworks required.

---

## License

MIT License, Copyright © 2025 Matthew Abbott

See [license.md](../GlassBoxAI-MLP/license.md) for details.

---
