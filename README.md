# GlassBoxAI-CNN

**Author:** Matthew Abbott (2025)

GlassBoxAI-CNN provides transparent, research-grade CUDA and OpenCL CNNs for deep learning with a focus on thorough command-line inspection.

---

## Table of Contents

- [Features](#features)
- [Module Overview](#module-overview)
- [Requirements](#requirements)
- [Quickstart: Compiling & Running](#quickstart-compiling--running)
- [CLI Usage and Help](#cli-usage-and-help)
  - [1. CUDA Command-Line CNN (`cnn.cu`)](#1-cuda-command-line-cnn-cnncu)
  - [2. OpenCL Command-Line CNN (`cnn_opencl.cpp`)](#2-opencl-command-line-cnn-cnn_openclcpp)
  - [3. CUDA Facade (`facaded_cnn.cu`)](#3-cuda-facade-facaded_cnncu)
  - [4. OpenCL Facade (`facaded_cnn_opencl.cpp`)](#4-opencl-facade-facaded_cnn_openclcpp)
    - [All Facade Introspection Options](#all-facade-introspection-options)
- [Architecture Notes](#architecture-notes)
- [License](#license)

---

## Features

- Pure dependency-free implementations in CUDA and OpenCL.
- **Two styles of interface:**  
  - **Core CLI** (`cnn.cu`, `cnn_opencl.cpp`): Lean CLI for model creation, training, and running/prediction. Straightforward for scripts and reproducible pipelines.
  - **Facade Introspection CLI** (`facaded_cnn.cu`, `facaded_cnn_opencl.cpp`): Rich CLI with deep model inspection, weight/bias editing, filter inspection, and debugging tools.
- Adam optimizer, ReLU and Sigmoid/Tanh activations, batch learning, gradient clipping.
- Max pooling, arbitrary channel/image/layer count, dropout, stable softmax.
- Model save/load from disk.
- CLI: All command-line arguments are parsed and displayed in help for reproducibility.
- Designed for hackability—inspect every major parameter, neuron, or filter from CLI or code.

---

## Module Overview

**4 modes:**

| Type   | Core CLI           | Facade/Introspectable CLI   |
|--------|--------------------|-----------------------------|
| CUDA   | `cnn.cu`           | `facaded_cnn.cu`            |
| OpenCL | `cnn_opencl.cpp`   | `facaded_cnn_opencl.cpp`    |

- **Core** = minimal CLI for script/production (create/train/predict/info).
- **Facade** = deep CLI for research/diagnostics/teaching.

---

## Requirements

- **CUDA** (`cnn.cu`, `facaded_cnn.cu`): NVIDIA GPU, CUDA Toolkit 11+, C++11, `nvcc`, `libcurand`
- **OpenCL** (`cnn_opencl.cpp`, `facaded_cnn_opencl.cpp`): OpenCL 1.2+ device, C++11, `g++` or `clang++`, `libOpenCL`
- *Optional*: CMake for integration
- Standard C++ build tools

---

## Quickstart: Compiling & Running

**CUDA:**
```bash
# core model CLI
nvcc -O2 -o cnn_cuda cnn.cu -lcurand

# facade CLI
nvcc -O2 -o facaded_cnn_cuda facaded_cnn.cu -lcurand
```

**OpenCL:**
```bash
# core CLI
g++ -O2 -std=c++14 -o cnn_opencl cnn_opencl.cpp -lOpenCL

# facade CLI
g++ -O2 -std=c++14 -o facaded_cnn_opencl facaded_cnn_opencl.cpp -lOpenCL
```

---

## CLI Usage and Help

### 1. CUDA Command-Line CNN (`cnn.cu`)
Minimal, scriptable CLI for learning/inference.

**Show help:**
```bash
./cnn_cuda help
```

#### Example Help Output (abridged):
```
CNN CUDA - Command-line Convolutional Neural Network
Matthew Abbott 2025

Commands:
  create   Create a new CNN model
  train    Train an existing model with data
  predict  Make predictions with a trained model
  info     Display model information
  help     Show this help message

Create Options:
  --input-w=N            Input width (required)
  --input-h=N            Input height (required)
  --input-c=N            Input channels (required)
  --conv=N,N,...         Conv filters (required)
  --kernels=N,N,...      Kernel sizes (required)
  --pools=N,N,...        Pool sizes (required)
  --fc=N,N,...           FC layer sizes (required)
  --output=N             Output layer size (required)
  --save=FILE            Save model to file (required)
  --lr=VALUE             Learning rate (default: 0.001)
  --hidden-act=TYPE      sigmoid|tanh|relu|linear (default: relu)
  --output-act=TYPE      sigmoid|tanh|relu|linear (default: linear)
  --loss=TYPE            mse|crossentropy (default: mse)
  --clip=VALUE           Gradient clipping (default: 5.0)

Examples:
  cnn_cuda create --input-w=28 --input-h=28 --input-c=1 --conv=32,64 --kernels=3,3 --pools=2,2 --fc=128 --output=10 --save=model.bin

  cnn_cuda train --model=model.bin --data=data.csv --epochs=50 --save=model_trained.bin
```

#### Argument Descriptions

| Argument           | Description                                                                  |
|--------------------|------------------------------------------------------------------------------|
| `create`           | Create a new CNN and save it to file                                         |
| `train`            | Train model with input/target pairs                                          |
| `predict`          | Output predictions using input and trained model                             |
| `info`             | Show model architecture/config                                               |
| `--input-w`        | Width of input image (int, required)                                         |
| `--input-h`        | Height of input image (int, required)                                        |
| `--input-c`        | Input channels (e.g. 1=gray, 3=RGB)                                          |
| `--conv`           | Comma-separated conv filter sizes (e.g., 32,64 for two conv layers)          |
| `--kernels`        | Comma-separated kernel sizes per conv layer (e.g., 3,3)                      |
| `--pools`          | Comma-separated max pooling sizes (e.g., 2,2)                                |
| `--fc`             | Comma-separated fully connected sizes (e.g., 128,64)                         |
| `--output`         | Output layer size (e.g., 10 for 10-class)                                    |
| `--save`           | File to save model                                                           |
| `--lr`             | Learning rate (default: 0.001)                                               |
| `--hidden-act`     | Hidden layer activation (sigmoid, tanh, relu, linear)                        |
| `--output-act`     | Output layer activation (sigmoid, tanh, relu, linear)                        |
| `--loss`           | Loss function type (mse, crossentropy)                                       |
| `--clip`           | Gradient clip value (default: 5.0)                                           |

---

### 2. OpenCL Command-Line CNN (`cnn_opencl.cpp`)

Identical logic and arguments to CUDA core, but using OpenCL backend.

**Show help:**
```bash
./cnn_opencl help
```

#### Example Help Output (abridged):

```
CNN OpenCL - GPU-accelerated Convolutional Neural Network

Commands:
  create   Create a new CNN model
  train    Train an existing model with data
  predict  Make predictions with a trained model
  info     Display model information
  help     Show this help message

Create Options:
  --input-w=N            Input width (required)
  --input-h=N            Input height (required)
  --input-c=N            Input channels (required)
  --conv=N,N,...         Conv filters (required)
  --kernels=N,N,...      Kernel sizes (required)
  --pools=N,N,...        Pool sizes (required)
  --fc=N,N,...           FC layer sizes (required)
  --output=N             Output layer size (required)
  --save=FILE            Save model to file (required)
  --lr=VALUE             Learning rate (default: 0.001)
  --hidden-act=TYPE      sigmoid|tanh|relu|linear (default: relu)
  --output-act=TYPE      sigmoid|tanh|relu|linear (default: linear)
  --loss=TYPE            mse|crossentropy (default: mse)
  --clip=VALUE           Gradient clipping (default: 5.0)

Examples:
  cnn_opencl create --input-w=28 --input-h=28 --input-c=1 --conv=32,64 --kernels=3,3 --pools=2,2 --fc=128 --output=10 --save=model.bin

  cnn_opencl train --model=model.bin --data=data.csv --epochs=50 --save=model_trained.bin
```

#### Argument Descriptions

(Identical to CUDA version above.)

---

### 3. CUDA Facade (`facaded_cnn.cu`)

All minimal commands plus dozens of facade CLI tools for deep model analysis!

**Show help:**
```bash
./facaded_cnn_cuda help
```

#### Example Help Output (abridged):

```
Facade CNN CUDA - GPU-accelerated CNN with Introspection
Matthew Abbott 2025

Commands:
  create       Create a new CNN model
  train        Train an existing model with data
  predict      Make predictions with a trained model
  info         Display model information
  get-filter   Get filter kernel values
  get-bias     Get filter bias value
  set-weight   Set individual weight value
  help         Show this help message

Create Options:
  --input-w=N            Input width (required)
  --input-h=N            Input height (required)
  --input-c=N            Input channels (required)
  --conv=N,N,...         Conv filters (required)
  --kernels=N,N,...      Kernel sizes (required)
  --pools=N,N,...        Pool sizes (required)
  --fc=N,N,...           FC layer sizes (required)
  --output=N             Output layer size (required)
  --save=FILE            Save model to file (required)
  --lr=VALUE             Learning rate (default: 0.001)
  --hidden-act=TYPE      sigmoid|tanh|relu|linear (default: relu)
  --output-act=TYPE      sigmoid|tanh|relu|linear (default: linear)
  --loss=TYPE            mse|crossentropy (default: mse)
  --clip=VALUE           Gradient clipping (default: 1.0)

Introspection Options:
  --layer=L              Layer index
  --filter=F             Filter index
  --channel=C            Channel index
  --h=H                  Height coordinate
  --w=W                  Width coordinate
  --value=V              Weight value (for set-weight)

Facade Introspection Commands:
  --get-feature-map=L,F        Get feature map for layer L, filter F
  --get-preactivation=L,F      Get pre-activation for layer L, filter F
  --get-kernel=L,F             Get kernel weights for layer L, filter F
  --get-bias=L,F               Get bias for layer L, filter F
  --get-filter-gradient=L,F    Get weight gradients for layer L, filter F
  --get-bias-gradient=L,F      Get bias gradient for layer L, filter F
  --get-pooling-indices=L,F    Get max pooling indices for layer L, filter F
  --get-flattened              Get flattened feature vector
  --get-logits                 Get raw logits from output layer
  --get-softmax                Get softmax probabilities
  --get-layer-stats=L          Get statistics for layer L
  --get-activation-hist=L      Get activation histogram for layer L
  --get-weight-hist=L          Get weight histogram for layer L
  --get-receptive-field=L      Get receptive field size at layer L
  --get-fc-weights=L           Get fully connected weights for layer L
  --get-fc-bias=L              Get fully connected bias for layer L
  --get-dropout-mask=L         Get dropout mask for layer L
  --add-filter=L,N             Add N new filters to layer L
  --get-num-filters=L          Get number of filters in layer L
  --set-bias=L,F,V             Set bias for layer L, filter F to value V
  --set-fc-bias=L,N,V          Set FC bias for layer L, neuron N to value V

Examples:
  facaded_cnn_cuda create --input-w=28 --input-h=28 --input-c=1 --conv=32,64 --kernels=3,3 --pools=2,2 --fc=128 --output=10 --save=model.bin
  facaded_cnn_cuda info --model=model.bin
  facaded_cnn_cuda get-filter --model=model.bin --layer=0 --filter=0 --channel=0
```

#### Description of Introspection/Facade Arguments

| Option / Command             | Description                                                   |
|------------------------------|---------------------------------------------------------------|
| `get-filter`                 | Print weights for filter F in layer L and channel C           |
| `get-bias`                   | Print bias for given filter                                   |
| `set-weight`                 | Set a specific weight value                                   |
| `--get-feature-map`          | Display the feature map for a layer/filter                    |
| `--get-kernel`               | Show kernel weights for a filter                              |
| `--get-preactivation`        | Display preactivation for a filter                            |
| `--get-flattened`            | Output the flattened representation after last pool/convlayer |
| `--get-logits`               | Print last-layer raw outputs                                  |
| `--get-softmax`              | Output last-layer softmax probabilities                       |
| `--set-bias`                 | Set a specific bias value for a filter                        |
| `--set-fc-bias`              | Set FC bias for neuron N in FC layer L                        |
| ... and more (see help)      |                                                               |

---

### 4. OpenCL Facade (`facaded_cnn_opencl.cpp`)

Same as CUDA facade, but runs on OpenCL backend.

**Show help:**
```bash
./facaded_cnn_opencl help
```

#### Example Help Output (abridged):

```
Facade CNN OpenCL - GPU-accelerated CNN with Introspection
Matthew Abbott 2025

Commands:
  create       Create a new CNN model
  train        Train an existing model with data
  predict      Make predictions with a trained model
  info         Display model information
  get-filter   Get filter kernel values
  get-bias     Get filter bias value
  set-weight   Set individual weight value
  help         Show this help message

Create Options:
  --input-w=N            Input width (required)
  --input-h=N            Input height (required)
  --input-c=N            Input channels (required)
  --conv=N,N,...         Conv filters (required)
  --kernels=N,N,...      Kernel sizes (required)
  --pools=N,N,...        Pool sizes (required)
  --fc=N,N,...           FC layer sizes (required)
  --output=N             Output layer size (required)
  --save=FILE            Save model to file (required)
  --lr=VALUE             Learning rate (default: 0.001)
  --hidden-act=TYPE      sigmoid|tanh|relu|linear (default: relu)
  --output-act=TYPE      sigmoid|tanh|relu|linear (default: linear)
  --loss=TYPE            mse|crossentropy (default: mse)
  --clip=VALUE           Gradient clipping (default: 1.0)

(Identical introspection/facade commands as CUDA version above)
```

---

## License

MIT License  
© 2025 Matthew Abbott

---
