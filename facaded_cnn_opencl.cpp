/*
 * MIT License
 *
 * Copyright (c) 2025 Matthew Abbott
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <random>
#include <CL/cl.h>

using namespace std;

const double EPSILON = 1e-8;
const double GRAD_CLIP = 1.0;

// ========== OpenCL Context ==========
class OpenCLContext {
public:
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    
    OpenCLContext() : context(nullptr), queue(nullptr) {
        Initialize();
    }
    
    ~OpenCLContext() {
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
    
    void Initialize() {
        cl_int err;
        clGetPlatformIDs(1, &platform, nullptr);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        queue = clCreateCommandQueue(context, device, 0, &err);
    }
};

static OpenCLContext gOpenCLContext;

// Utility functions for string conversion
string FloatToStr(double V) {
    ostringstream oss;
    oss << V;
    return oss.str();
}

string IntToStr(int V) {
    return to_string(V);
}

// Common array types for storing weights, activations, filters, etc.
typedef vector<double> Darray;
typedef vector<vector<double>> D2array;
typedef vector<vector<vector<double>>> D3array;
typedef vector<vector<vector<vector<double>>>> D4array;
typedef vector<int> IntArray;

// Index used for pooling -- points to (X,Y) max in a pooling window
struct TPoolIndex {
    int X, Y;
};
typedef vector<vector<vector<TPoolIndex>>> TPoolIndexArray;

// Useful for summarizing statistics about a layer's activations/weights
struct TLayerStats {
    double Mean;
    double StdDev;
    double Min;
    double Max;
    int Count;
};

// Configuration for a given network layer (introspection/debug)
struct TLayerConfig {
    string LayerType;           // "conv", "pool", or "fc"
    int FilterCount;            // For conv:  how many filters
    int KernelSize;
    int Stride;
    int Padding;
    int InputChannels;
    int OutputWidth;
    int OutputHeight;
    int PoolSize;               // For pool layers
    int NeuronCount;            // For fully connected
    int InputSize;              // For fully connected:  neurons' input size
};

// For computing receptive fields (advanced visualization)
struct TReceptiveField {
    int StartX, EndX;
    int StartY, EndY;
    IntArray Channels;          // Which input channels are connected
};

// For arbitrary string tags/metadata on net filters/layers
struct TAttributeEntry {
    string Key;
    string Value;
};
typedef vector<TAttributeEntry> TAttributeArray;

// A single image (input), as 3D array [channels][height][width]
struct TImageData {
    int Width;
    int Height;
    int Channels;
    D3array Data;               // Data[channel][row][col]
};

// A single convolution filter (including gradient and optimizer state)
struct TConvFilter {
    D3array Weights;            // Main weights [Channel][H][W]
    double Bias;
    D3array WeightsM;           // Adam optimizer moment-1
    D3array WeightsV;           // Adam optimizer moment-2
    D3array WeightGrads;        // Gradients dL/dW
    double BiasGrad;
    double BiasM;               // For bias Adam optimizer
    double BiasV;
    
    cl_mem WeightsBuffer;
    cl_mem WeightGradsBuffer;
};

// A convolution layer holds several filters & their output/activations
struct TConvLayer {
    vector<TConvFilter> Filters;     // List of filters in this layer
    D3array OutputMaps;              // Activations [Filter][H][W]
    D3array PreActivation;           // Z maps before ReLU [Filter][H][W]
    D3array InputCache;              // Cached input for backwards
    D3array PaddedInput;             // Input w/padding (for backprop)
    int Stride;
    int Padding;
    int KernelSize;
    int InputChannels;
    
    cl_mem OutputBuffer;
    cl_mem InputBuffer;
};

// A pooling layer records window size and related caches
struct TPoolingLayer {
    int PoolSize;
    int Stride;
    D3array OutputMaps;              // Output after pooling
    D3array InputCache;              // What was pooled over
    TPoolIndexArray MaxIndices;      // Where was the max for each win
};

// A neuron in a fully connected (dense) layer
struct TNeuron {
    Darray Weights;
    double Bias;
    double Output;                   // Activation value
    double PreActivation;            // Z before activation
    double Error;                    // Gradient w.r.t activation for backprop
    double DropoutMask;              // 0 or scaled 1 (for dropout regularization)
    Darray WeightsM;                 // Adam moment-1
    Darray WeightsV;                 // Adam moment-2
    double BiasM;
    double BiasV;
};

// Fully connected layer:  array of neurons, stores input for backprop
struct TFullyConnectedLayer {
    vector<TNeuron> Neurons;
    Darray InputCache;
};

// Batch normalization parameters and state (for advanced usage)
struct TBatchNormParams {
    Darray Gamma;
    Darray Beta;
    Darray RunningMean;
    Darray RunningVar;
    bool Enabled;
};

/*--------------------------------------------------------------------------
 MAIN CLASS: TCNNFacade
 All properties and methods attached to this object.
 -------------------------------------------------------------------------*/
class TCNNFacade {
private:
    // Hyperparameters, optimizer and state
    double LearningRate;
    double DropoutRate;
    double Beta1, Beta2;
    int AdamT;
    bool IsTraining;
    double GradientClip;
    int OutputSize;
    int ActivationType;
    int OutputActivationType;
    int LossType;

    // Input dimensions
    int FInputWidth, FInputHeight, FInputChannels;

    // Layers of the network, in order of computation
    vector<TConvLayer> ConvLayers;
    vector<TPoolingLayer> PoolLayers;
    vector<TFullyConnectedLayer> FullyConnectedLayers;
    TFullyConnectedLayer OutputLayer;

    // Sizes for internal bookkeeping
    int FlattenedSize;
    Darray FFlattenedFeatures;
    int LastConvHeight, LastConvWidth, LastConvChannels;

    // Stubs for batch (mini-batch) functionality, not complete in base version
    vector<D4array> FBatchActivations;
    vector<TBatchNormParams> FBatchNormParams;
    vector<vector<TAttributeArray>> FFilterAttributes;

    /*--------------------- INTERNAL UTILITY METHODS ----------------------*/
    double ReLU(double x);
    double ReLUDerivative(double x);
    Darray Softmax(const Darray& Logits);
    double CrossEntropyLoss(const Darray& Predicted, const Darray& Target);
    double ClipGrad(double x);
    bool IsFiniteNum(double x);
    double Clamp(double x, double MinVal, double MaxVal);
    D3array Pad3D(const D3array& Input, int Padding);
    bool ValidateInput(const TImageData& Image);

    // Layer/parameter initializers
    void InitializeConvLayer(TConvLayer& Layer, int NumFilters, int InputChannels, int KernelSize, int Stride, int Padding);
    void InitializePoolLayer(TPoolingLayer& Layer, int PoolSize, int Stride);
    void InitializeFCLayer(TFullyConnectedLayer& Layer, int NumNeurons, int NumInputs);

    // Forward/Backward steps for each layer type
    void ConvForward(TConvLayer& Layer, const D3array& Input, int InputWidth, int InputHeight);
    void PoolForward(TPoolingLayer& Layer, const D3array& Input, int InputWidth, int InputHeight);
    void FlattenFeatures(const D3array& Input, int InputWidth, int InputHeight, int InputChannels);
    void FCForward(TFullyConnectedLayer& Layer, const Darray& Input);

    D3array ConvBackward(TConvLayer& Layer, const D3array& Grad);
    D3array PoolBackward(TPoolingLayer& Layer, const D3array& Grad);
    Darray FCBackward(TFullyConnectedLayer& Layer, const Darray& Grad, bool IsOutputLayer);
    D3array UnflattenGradient(const Darray& Grad);

    void UpdateWeights();
    void ApplyDropout(TFullyConnectedLayer& Layer);

    /* JSON Helper functions */
    string Array1DToJSON(const Darray& Arr);
    string Array2DToJSON(const D2array& Arr);
    string Array3DToJSON(const D3array& Arr);

public:
    /*================= CONSTRUCTION / TEARDOWN =================*/
    TCNNFacade(int InputWidth, int InputHeight, int InputChannels,
               const vector<int>& ConvFilters, const vector<int>& KernelSizes,
               const vector<int>& PoolSizes, const vector<int>& FCLayerSizes,
               int OutputSize, double ALearningRate = 0.001, double ADropoutRate = 0.25);
    ~TCNNFacade();

    /*================== MAIN FUNCTIONALITY =====================*/
    Darray Predict(TImageData& Image);
    double TrainStep(TImageData& Image, const Darray& Target);
    void SaveModel(const string& Filename);
    void LoadModel(const string& Filename);

    /* JSON serialization methods */
    void SaveModelToJSON(const string& Filename);
    void LoadModelFromJSON(const string& Filename);

    /* ONNX Export/Import */
    void ExportToONNX(const string& Filename);
    static TCNNFacade* ImportFromONNX(const string& Filename);

    /* Input dimension getters */
    int GetInputWidth();
    int GetInputHeight();
    int GetInputChannels();

    /* Training parameter getters */
    int GetOutputSize();
    double GetLearningRate();
    double GetGradientClip();
    string GetActivationTypeString();
    string GetOutputActivationTypeString();
    int GetLossType();

    /* Stage 1: Feature Map Access */
    D2array GetFeatureMap(int LayerIdx, int FilterIdx);
    void SetFeatureMap(int LayerIdx, int FilterIdx, const D2array& Map);

    /* Stage 1: Pre-Activation Access */
    D2array GetPreActivation(int LayerIdx, int FilterIdx);
    void SetPreActivation(int LayerIdx, int FilterIdx, const D2array& Map);

    /* Stage 2: Kernel/Filter Access */
    D2array GetKernel(int LayerIdx, int FilterIdx, int ChannelIdx);
    void SetKernel(int LayerIdx, int FilterIdx, int ChannelIdx, const D2array& KernelArray);
    double GetBias(int LayerIdx, int FilterIdx);
    void SetBias(int LayerIdx, int FilterIdx, double Value);

    /* Stage 3: Batch Activations */
    D4array GetBatchActivations(int LayerIdx);
    void SetBatchActivations(int LayerIdx, const D4array& BatchTensor);

    /* Stage 4: Pooling and Dropout States */
    D2array GetPoolingIndices(int LayerIdx, int FilterIdx);
    Darray GetDropoutMask(int LayerIdx);
    void SetDropoutMask(int LayerIdx, const Darray& Mask);

    /* Stage 5: Gradients */
    D2array GetFilterGradient(int LayerIdx, int FilterIdx, int ChannelIdx);
    double GetBiasGradient(int LayerIdx, int FilterIdx);
    double GetActivationGradient(int LayerIdx, int FilterIdx, int Y, int X);
    D3array GetOptimizerState(int LayerIdx, int FilterIdx, const string& Param);

    /* Stage 6: Flattened Features */
    Darray GetFlattenedFeatures();
    void SetFlattenedFeatures(const Darray& Vector);

    /* Stage 7: Output Layer */
    Darray GetLogits();
    Darray GetSoftmax();

    /* Stage 8: Layer Config */
    TLayerConfig GetLayerConfig(int LayerIdx);
    int GetNumLayers();
    int GetNumConvLayers();
    int GetNumFCLayers();
    int GetNumFilters(int LayerIdx);

    /* Stage 9: Saliency and Deconv */
    D2array GetSaliencyMap(int LayerIdx, int FilterIdx, int InputIdx);
    D3array GetDeconv(int LayerIdx, int FilterIdx, bool UpToInput);

    /* Stage 10: Structural Mutations */
    void AddFilter(int LayerIdx, const D3array& Params);
    void RemoveFilter(int LayerIdx, int FilterIdx);
    void AddConvLayer(int Position, int NumFilters, int KernelSize, int Stride, int Padding);
    void RemoveLayer(int LayerIdx);

    /* Stage 11: Statistics */
    TLayerStats GetLayerStats(int LayerIdx);
    Darray GetActivationHistogram(int LayerIdx, int NumBins = 50);
    Darray GetWeightHistogram(int LayerIdx, int NumBins = 50);

    /* Stage 12: Receptive Field */
    TReceptiveField GetReceptiveField(int LayerIdx, int FeatureIdx, int Y, int X);

    /* Stage 13: Batch Norm */
    TBatchNormParams GetBatchNormParams(int LayerIdx);
    void SetBatchNormParams(int LayerIdx, const TBatchNormParams& Params);

    /* Stage 14: Attributes */
    void SetFilterAttribute(int LayerIdx, int FilterIdx, const string& Key, const string& Value);
    string GetFilterAttribute(int LayerIdx, int FilterIdx, const string& Key);

    /* Training state */
    void SetTrainingMode(bool Training);
    bool GetTrainingMode();
};

/* Helper functions */

/* JSON serialization helper functions */
string TCNNFacade::Array1DToJSON(const Darray& Arr) {
    string Result = "[";
    for (size_t i = 0; i < Arr.size(); i++) {
        if (i > 0) Result += ",";
        Result += FloatToStr(Arr[i]);
    }
    Result += "]";
    return Result;
}

string TCNNFacade::Array2DToJSON(const D2array& Arr) {
    string Result = "[";
    for (size_t i = 0; i < Arr.size(); i++) {
        if (i > 0) Result += ",";
        Result += Array1DToJSON(Arr[i]);
    }
    Result += "]";
    return Result;
}

string TCNNFacade::Array3DToJSON(const D3array& Arr) {
    string Result = "[";
    for (size_t i = 0; i < Arr.size(); i++) {
        if (i > 0) Result += ",";
        Result += Array2DToJSON(Arr[i]);
    }
    Result += "]";
    return Result;
}

bool TCNNFacade::IsFiniteNum(double x) {
    return !isnan(x) && !isinf(x);
}

double TCNNFacade::Clamp(double x, double MinVal, double MaxVal) {
    if (x < MinVal) return MinVal;
    else if (x > MaxVal) return MaxVal;
    else return x;
}

double TCNNFacade::ClipGrad(double x) {
    if (!IsFiniteNum(x)) return 0.0;
    else return Clamp(x, -GRAD_CLIP, GRAD_CLIP);
}

double TCNNFacade::ReLU(double x) {
    if (x > 0) return x;
    else return 0.0;
}

double TCNNFacade::ReLUDerivative(double x) {
    if (x > 0) return 1.0;
    else return 0.0;
}

Darray TCNNFacade::Softmax(const Darray& Logits) {
    Darray Result(Logits.size());
    double MaxVal = -1e308;

    for (size_t i = 0; i < Logits.size(); i++) {
        if (IsFiniteNum(Logits[i]) && (Logits[i] > MaxVal)) {
            MaxVal = Logits[i];
        }
    }
    if (!IsFiniteNum(MaxVal)) MaxVal = 0;

    double Sum = 0;
    for (size_t i = 0; i < Logits.size(); i++) {
        double ExpVal;
        if (IsFiniteNum(Logits[i]))
            ExpVal = exp(Clamp(Logits[i] - MaxVal, -500, 500));
        else
            ExpVal = exp(0);
        Result[i] = ExpVal;
        Sum += ExpVal;
    }

    if ((Sum <= 0) || (!IsFiniteNum(Sum))) Sum = 1;
    for (size_t i = 0; i < Result.size(); i++) {
        Result[i] = Clamp(Result[i] / Sum, 1e-15, 1 - 1e-15);
    }

    return Result;
}

double TCNNFacade::CrossEntropyLoss(const Darray& Predicted, const Darray& Target) {
    double Result = 0;
    for (size_t i = 0; i < Target.size(); i++) {
        if (Target[i] > 0) {
            double P = Clamp(Predicted[i], 1e-15, 1 - 1e-15);
            Result -= Target[i] * log(P);
        }
    }
    if (!IsFiniteNum(Result)) Result = 0;
    return Result;
}

D3array TCNNFacade::Pad3D(const D3array& Input, int Padding) {
    if (Padding == 0) return Input;

    int Channels = Input.size();
    int Height = Input[0].size();
    int Width = Input[0][0].size();

    D3array Result(Channels, D2array(Height + 2*Padding, Darray(Width + 2*Padding, 0.0)));

    for (int c = 0; c < Channels; c++) {
        for (int h = 0; h < Height + 2*Padding; h++) {
            for (int w = 0; w < Width + 2*Padding; w++) {
                int SrcH = h - Padding;
                int SrcW = w - Padding;
                if ((SrcH >= 0) && (SrcH < Height) && (SrcW >= 0) && (SrcW < Width)) {
                    Result[c][h][w] = Input[c][SrcH][SrcW];
                } else {
                    Result[c][h][w] = 0;
                }
            }
        }
    }
    return Result;
}

bool TCNNFacade::ValidateInput(const TImageData& Image) {
    if (Image.Data.empty() || (int)Image.Data.size() != Image.Channels) return false;

    for (int c = 0; c < Image.Channels; c++) {
        if (Image.Data[c].empty() || (int)Image.Data[c].size() != Image.Height) return false;
        for (int h = 0; h < Image.Height; h++) {
            if (Image.Data[c][h].empty() || (int)Image.Data[c][h].size() != Image.Width) return false;
            for (int w = 0; w < Image.Width; w++) {
                if (!IsFiniteNum(Image.Data[c][h][w])) return false;
            }
        }
    }
    return true;
}

TCNNFacade::TCNNFacade(int InputWidth, int InputHeight, int InputChannels,
    const vector<int>& ConvFilters, const vector<int>& KernelSizes,
    const vector<int>& PoolSizes, const vector<int>& FCLayerSizes,
    int OutputSize, double ALearningRate, double ADropoutRate) {

    LearningRate = ALearningRate;
    DropoutRate = ADropoutRate;
    Beta1 = 0.9;
    Beta2 = 0.999;
    AdamT = 0;
    IsTraining = false;
    GradientClip = 5.0;
    this->OutputSize = OutputSize;
    ActivationType = 2; // ReLU
    OutputActivationType = 3; // Linear
    LossType = 0; // MSE

    /* Store input dimensions */
    FInputWidth = InputWidth;
    FInputHeight = InputHeight;
    FInputChannels = InputChannels;

    int CurrentWidth = InputWidth;
    int CurrentHeight = InputHeight;
    int CurrentChannels = InputChannels;

    ConvLayers.resize(ConvFilters.size());
    PoolLayers.resize(PoolSizes.size());

    for (size_t i = 0; i < ConvFilters.size(); i++) {
        int KernelPadding = KernelSizes[i] / 2;
        InitializeConvLayer(ConvLayers[i], ConvFilters[i], CurrentChannels,
                           KernelSizes[i], 1, KernelPadding);
        CurrentWidth = (CurrentWidth - KernelSizes[i] + 2 * KernelPadding) / 1 + 1;
        CurrentHeight = (CurrentHeight - KernelSizes[i] + 2 * KernelPadding) / 1 + 1;
        CurrentChannels = ConvFilters[i];

        if (i < PoolSizes.size()) {
            InitializePoolLayer(PoolLayers[i], PoolSizes[i], PoolSizes[i]);
            CurrentWidth = CurrentWidth / PoolSizes[i];
            CurrentHeight = CurrentHeight / PoolSizes[i];
        }
    }

    LastConvWidth = CurrentWidth;
    LastConvHeight = CurrentHeight;
    LastConvChannels = CurrentChannels;
    FlattenedSize = CurrentWidth * CurrentHeight * CurrentChannels;

    FullyConnectedLayers.resize(FCLayerSizes.size());
    int NumInputs = FlattenedSize;

    for (size_t i = 0; i < FCLayerSizes.size(); i++) {
        InitializeFCLayer(FullyConnectedLayers[i], FCLayerSizes[i], NumInputs);
        NumInputs = FCLayerSizes[i];
    }

    InitializeFCLayer(OutputLayer, OutputSize, NumInputs);
}

TCNNFacade::~TCNNFacade() {
    // Release GPU buffers
    for (size_t i = 0; i < ConvLayers.size(); i++) {
        if (ConvLayers[i].OutputBuffer) clReleaseMemObject(ConvLayers[i].OutputBuffer);
        if (ConvLayers[i].InputBuffer) clReleaseMemObject(ConvLayers[i].InputBuffer);
        for (size_t f = 0; f < ConvLayers[i].Filters.size(); f++) {
            if (ConvLayers[i].Filters[f].WeightsBuffer) clReleaseMemObject(ConvLayers[i].Filters[f].WeightsBuffer);
            if (ConvLayers[i].Filters[f].WeightGradsBuffer) clReleaseMemObject(ConvLayers[i].Filters[f].WeightGradsBuffer);
        }
    }
}

Darray TCNNFacade::Predict(TImageData& Image) {
    if (!ValidateInput(Image)) {
        Darray Result(OutputLayer.Neurons.size(), 0.0);
        return Result;
    }

    D3array CurrentMaps = Image.Data;
    int CurrentWidth = Image.Width;
    int CurrentHeight = Image.Height;

    for (size_t i = 0; i < ConvLayers.size(); i++) {
        ConvForward(ConvLayers[i], CurrentMaps, CurrentWidth, CurrentHeight);
        CurrentWidth = ConvLayers[i].OutputMaps[0][0].size();
        CurrentHeight = ConvLayers[i].OutputMaps[0].size();

        if (i < PoolLayers.size()) {
            PoolForward(PoolLayers[i], ConvLayers[i].OutputMaps, CurrentWidth, CurrentHeight);
            CurrentMaps = PoolLayers[i].OutputMaps;
            CurrentWidth = PoolLayers[i].OutputMaps[0][0].size();
            CurrentHeight = PoolLayers[i].OutputMaps[0].size();
        } else {
            CurrentMaps = ConvLayers[i].OutputMaps;
        }
    }

    FlattenFeatures(CurrentMaps, CurrentWidth, CurrentHeight, CurrentMaps.size());
    Darray LayerInput = FFlattenedFeatures;

    for (size_t i = 0; i < FullyConnectedLayers.size(); i++) {
        ApplyDropout(FullyConnectedLayers[i]);
        FCForward(FullyConnectedLayers[i], LayerInput);
        LayerInput.resize(FullyConnectedLayers[i].Neurons.size());
        for (size_t j = 0; j < FullyConnectedLayers[i].Neurons.size(); j++) {
            LayerInput[j] = FullyConnectedLayers[i].Neurons[j].Output;
        }
    }

    OutputLayer.InputCache = LayerInput;
    Darray Logits(OutputLayer.Neurons.size());
    for (size_t i = 0; i < OutputLayer.Neurons.size(); i++) {
        double Sum = OutputLayer.Neurons[i].Bias;
        for (size_t j = 0; j < LayerInput.size(); j++) {
            Sum += LayerInput[j] * OutputLayer.Neurons[i].Weights[j];
        }
        if (!IsFiniteNum(Sum)) Sum = 0;
        OutputLayer.Neurons[i].PreActivation = Sum;
        Logits[i] = Sum;
    }

    Darray Result = Softmax(Logits);
    for (size_t i = 0; i < OutputLayer.Neurons.size(); i++) {
        OutputLayer.Neurons[i].Output = Result[i];
    }

    return Result;
}

double TCNNFacade::TrainStep(TImageData& Image, const Darray& Target) {
    if (!ValidateInput(Image)) {
        return 0;
    }

    IsTraining = true;
    Darray Prediction = Predict(Image);

    Darray OutputGrad(OutputLayer.Neurons.size());
    for (size_t i = 0; i < OutputLayer.Neurons.size(); i++) {
        OutputGrad[i] = OutputLayer.Neurons[i].Output - Target[i];
    }

    Darray FCGrad = FCBackward(OutputLayer, OutputGrad, true);

    for (int i = FullyConnectedLayers.size() - 1; i >= 0; i--) {
        FCGrad = FCBackward(FullyConnectedLayers[i], FCGrad, false);
    }

    D3array ConvGrad = UnflattenGradient(FCGrad);

    for (int i = ConvLayers.size() - 1; i >= 0; i--) {
        if (i < (int)PoolLayers.size()) {
            ConvGrad = PoolBackward(PoolLayers[i], ConvGrad);
        }
        ConvGrad = ConvBackward(ConvLayers[i], ConvGrad);
    }

    UpdateWeights();
    return CrossEntropyLoss(Prediction, Target);
}

void TCNNFacade::SaveModelToJSON(const string& Filename) {
    vector<string> JSON;

    JSON.push_back("{");
    JSON.push_back("  \"input_width\": " + IntToStr(GetInputWidth()) + ",");
    JSON.push_back("  \"input_height\": " + IntToStr(GetInputHeight()) + ",");
    JSON.push_back("  \"input_channels\": " + IntToStr(GetInputChannels()) + ",");
    JSON.push_back("  \"output_size\": " + IntToStr(OutputLayer.Neurons.size()) + ",");
    JSON.push_back("  \"conv_filters\": [");
    for (size_t i = 0; i < ConvLayers.size(); i++) {
        string line = "    " + IntToStr(ConvLayers[i].Filters.size());
        if (i < ConvLayers.size() - 1) line += ",";
        JSON.push_back(line);
    }
    JSON.push_back("  ],");
    JSON.push_back("  \"kernel_sizes\": [");
    for (size_t i = 0; i < ConvLayers.size(); i++) {
        string line = "    " + IntToStr(ConvLayers[i].KernelSize);
        if (i < ConvLayers.size() - 1) line += ",";
        JSON.push_back(line);
    }
    JSON.push_back("  ],");
    JSON.push_back("  \"pool_sizes\": [");
    for (size_t i = 0; i < PoolLayers.size(); i++) {
        string line = "    " + IntToStr(PoolLayers[i].PoolSize);
        if (i < PoolLayers.size() - 1) line += ",";
        JSON.push_back(line);
    }
    JSON.push_back("  ],");
    JSON.push_back("  \"fc_layer_sizes\": [");
    for (size_t i = 0; i < FullyConnectedLayers.size(); i++) {
        string line = "    " + IntToStr(FullyConnectedLayers[i].Neurons.size());
        if (i < FullyConnectedLayers.size() - 1) line += ",";
        JSON.push_back(line);
    }
    JSON.push_back("  ],");
    JSON.push_back("  \"learning_rate\": " + FloatToStr(LearningRate) + ",");
    JSON.push_back("  \"dropout_rate\": " + FloatToStr(DropoutRate) + ",");
    JSON.push_back("  \"activation\": \"relu\",");
    JSON.push_back("  \"output_activation\": \"linear\",");
    JSON.push_back("  \"loss_type\": \"mse\",");
    JSON.push_back("  \"conv_layers\": [");

    for (size_t i = 0; i < ConvLayers.size(); i++) {
        JSON.push_back("    {");
        JSON.push_back("      \"filters\": [");
        for (size_t f = 0; f < ConvLayers[i].Filters.size(); f++) {
            JSON.push_back("        {");
            JSON.push_back("          \"bias\": " + FloatToStr(ConvLayers[i].Filters[f].Bias) + ",");
            JSON.push_back("          \"weights\": " + Array3DToJSON(ConvLayers[i].Filters[f].Weights));
            string line = "        }";
            if (f < ConvLayers[i].Filters.size() - 1) line += ",";
            JSON.push_back(line);
        }
        JSON.push_back("      ]");
        string line = "    }";
        if (i < ConvLayers.size() - 1) line += ",";
        JSON.push_back(line);
    }
    JSON.push_back("  ],");

    JSON.push_back("  \"pool_layers\": [");
    for (size_t i = 0; i < PoolLayers.size(); i++) {
        string line = "    {\"poolSize\": " + IntToStr(PoolLayers[i].PoolSize) + "}";
        if (i < PoolLayers.size() - 1) line += ",";
        JSON.push_back(line);
    }
    JSON.push_back("  ],");

    JSON.push_back("  \"fc_layers\": [");
    for (size_t i = 0; i < FullyConnectedLayers.size(); i++) {
        JSON.push_back("    {");
        JSON.push_back("      \"neurons\": [");
        for (size_t j = 0; j < FullyConnectedLayers[i].Neurons.size(); j++) {
            JSON.push_back("        {");
            JSON.push_back("          \"bias\": " + FloatToStr(FullyConnectedLayers[i].Neurons[j].Bias) + ",");
            JSON.push_back("          \"weights\": " + Array1DToJSON(FullyConnectedLayers[i].Neurons[j].Weights));
            string line = "        }";
            if (j < FullyConnectedLayers[i].Neurons.size() - 1) line += ",";
            JSON.push_back(line);
        }
        JSON.push_back("      ]");
        string line = "    }";
        if (i < FullyConnectedLayers.size() - 1) line += ",";
        JSON.push_back(line);
    }
    JSON.push_back("  ],");

    JSON.push_back("  \"output_layer\": {");
    JSON.push_back("    \"neurons\": [");
    for (size_t j = 0; j < OutputLayer.Neurons.size(); j++) {
        JSON.push_back("      {");
        JSON.push_back("        \"bias\": " + FloatToStr(OutputLayer.Neurons[j].Bias) + ",");
        JSON.push_back("        \"weights\": " + Array1DToJSON(OutputLayer.Neurons[j].Weights));
        string line = "      }";
        if (j < OutputLayer.Neurons.size() - 1) line += ",";
        JSON.push_back(line);
    }
    JSON.push_back("    ]");
    JSON.push_back("  }");

    JSON.push_back("}");

    ofstream outFile(Filename);
    for (const string& line : JSON) {
        outFile << line << endl;
    }
    outFile.close();

    cout << "Model saved to: " << Filename << endl;
}

// JSON Helper Functions
int ExtractIntFromJSON(const string& JSONStr, const string& FieldName) {
    size_t P = JSONStr.find("\"" + FieldName + "\"");
    if (P == string::npos) return 0;
    P = JSONStr.find(':', P);
    if (P == string::npos) return 0;
    P = P + 1;
    while ((P < JSONStr.length()) && (JSONStr[P] == ' ' || JSONStr[P] == '\t' ||
                                       JSONStr[P] == '\n' || JSONStr[P] == '\r')) P++;
    size_t EndP = P;
    while ((EndP < JSONStr.length()) && (isdigit(JSONStr[EndP]) || JSONStr[EndP] == '-')) EndP++;
    string Value = JSONStr.substr(P, EndP - P);
    try { return stoi(Value); } catch (...) { return 0; }
}

double ExtractDoubleFromJSON(const string& JSONStr, const string& FieldName) {
    size_t P = JSONStr.find("\"" + FieldName + "\"");
    if (P == string::npos) return 0.0;
    P = JSONStr.find(':', P);
    if (P == string::npos) return 0.0;
    P = P + 1;
    while ((P < JSONStr.length()) && (JSONStr[P] == ' ' || JSONStr[P] == '\t' ||
                                       JSONStr[P] == '\n' || JSONStr[P] == '\r')) P++;
    size_t EndP = P;
    while ((EndP < JSONStr.length()) && (isdigit(JSONStr[EndP]) || JSONStr[EndP] == '-' ||
                                          JSONStr[EndP] == '.' || JSONStr[EndP] == 'e' || JSONStr[EndP] == 'E')) EndP++;
    string Value = JSONStr.substr(P, EndP - P);
    try { return stod(Value); } catch (...) { return 0.0; }
}

string ExtractStringFromJSON(const string& JSONStr, const string& FieldName) {
    size_t P = JSONStr.find("\"" + FieldName + "\"");
    if (P == string::npos) return "";
    P = JSONStr.find(':', P);
    if (P == string::npos) return "";
    P = JSONStr.find('"', P);
    if (P == string::npos) return "";
    P++;
    size_t EndP = P;
    while ((EndP < JSONStr.length()) && (JSONStr[EndP] != '"')) EndP++;
    return JSONStr.substr(P, EndP - P);
}

int ExtractIntFromJSONArray(const string& JSONStr, const string& ArrayName, int Index, const string& FieldName) {
    size_t ArrayPos = JSONStr.find("\"" + ArrayName + "\"");
    if (ArrayPos == string::npos) return 0;
    ArrayPos = JSONStr.find('[', ArrayPos);
    if (ArrayPos == string::npos) return 0;
    int Count = 0;
    size_t ElementPos = ArrayPos + 1;
    while ((Count < Index) && (ElementPos < JSONStr.length())) {
        if (JSONStr[ElementPos] == '{') Count++;
        ElementPos++;
    }
    if (Count != Index) return 0;
    string SubStr = JSONStr.substr(ElementPos);
    size_t FieldPos = SubStr.find("\"" + FieldName + "\"");
    if (FieldPos == string::npos) return 0;
    FieldPos = ElementPos + FieldPos;
    size_t P = JSONStr.find(':', FieldPos);
    if (P == string::npos) return 0;
    P = P + 1;
    while ((P < JSONStr.length()) && (JSONStr[P] == ' ' || JSONStr[P] == '\t' ||
                                       JSONStr[P] == '\n' || JSONStr[P] == '\r')) P++;
    size_t EndP = P;
    while ((EndP < JSONStr.length()) && (isdigit(JSONStr[EndP]) || JSONStr[EndP] == '-')) EndP++;
    string Value = JSONStr.substr(P, EndP - P);
    try { return stoi(Value); } catch (...) { return 0; }
}

double ExtractDoubleFromJSONArray(const string& JSONStr, const string& ArrayName, int Index, const string& FieldName) {
    size_t ArrayPos = JSONStr.find("\"" + ArrayName + "\"");
    if (ArrayPos == string::npos) return 0.0;
    ArrayPos = JSONStr.find('[', ArrayPos);
    if (ArrayPos == string::npos) return 0.0;
    int Count = 0;
    size_t ElementPos = ArrayPos + 1;
    while ((Count < Index) && (ElementPos < JSONStr.length())) {
        if (JSONStr[ElementPos] == '{') Count++;
        ElementPos++;
    }
    if (Count != Index) return 0.0;
    string SubStr = JSONStr.substr(ElementPos);
    size_t FieldPos = SubStr.find("\"" + FieldName + "\"");
    if (FieldPos == string::npos) return 0.0;
    FieldPos = ElementPos + FieldPos;
    size_t P = JSONStr.find(':', FieldPos);
    if (P == string::npos) return 0.0;
    P = P + 1;
    while ((P < JSONStr.length()) && (JSONStr[P] == ' ' || JSONStr[P] == '\t' ||
                                       JSONStr[P] == '\n' || JSONStr[P] == '\r')) P++;
    size_t EndP = P;
    while ((EndP < JSONStr.length()) && (isdigit(JSONStr[EndP]) || JSONStr[EndP] == '-' ||
                                          JSONStr[EndP] == '.' || JSONStr[EndP] == 'e' || JSONStr[EndP] == 'E')) EndP++;
    string Value = JSONStr.substr(P, EndP - P);
    try { return stod(Value); } catch (...) { return 0.0; }
}

void LoadWeights1DFromJSONOutputLayer(const string& JSONStr, int NeuronIndex, Darray& Arr) {
    size_t P = JSONStr.find("\"output_layer\"");
    if (P == string::npos) return;
    P = JSONStr.find("\"neurons\"", P);
    if (P == string::npos) return;
    P = JSONStr.find('[', P);
    if (P == string::npos) return;
    int Count = 0;
    size_t ElementPos = P + 1;
    while ((Count < NeuronIndex) && (ElementPos < JSONStr.length())) {
        if (JSONStr[ElementPos] == '{') Count++;
        ElementPos++;
    }
    if (Count != NeuronIndex) return;
    string SubStr = JSONStr.substr(ElementPos);
    size_t FieldPos = SubStr.find("\"weights\"");
    if (FieldPos == string::npos) return;
    P = ElementPos + FieldPos;
    size_t ArrayStartPos = JSONStr.find('[', P);
    if (ArrayStartPos == string::npos) return;
    Count = 1;
    size_t ArrayEndPos = ArrayStartPos + 1;
    while ((Count > 0) && (ArrayEndPos < JSONStr.length())) {
        if (JSONStr[ArrayEndPos] == '[') Count++;
        else if (JSONStr[ArrayEndPos] == ']') Count--;
        ArrayEndPos++;
    }
    Arr.clear();
    size_t CurrentPos = ArrayStartPos + 1;
    while ((CurrentPos < ArrayEndPos) && (JSONStr[CurrentPos] != ']')) {
        if (isdigit(JSONStr[CurrentPos]) || JSONStr[CurrentPos] == '-' || JSONStr[CurrentPos] == '.') {
            size_t NumPos = CurrentPos;
            while ((NumPos < JSONStr.length()) &&
                   (isdigit(JSONStr[NumPos]) || JSONStr[NumPos] == '-' ||
                    JSONStr[NumPos] == '.' || JSONStr[NumPos] == 'e' || JSONStr[NumPos] == 'E')) {
                NumPos++;
            }
            string Value = JSONStr.substr(CurrentPos, NumPos - CurrentPos);
            try { Arr.push_back(stod(Value)); } catch (...) { Arr.push_back(0.0); }
            CurrentPos = NumPos;
        } else {
            CurrentPos++;
        }
    }
}

void TCNNFacade::LoadModelFromJSON(const string& Filename) {
    ifstream inFile(Filename);
    if (!inFile.is_open()) {
        cout << "Failed to open: " << Filename << endl;
        return;
    }
    
    stringstream buffer;
    buffer << inFile.rdbuf();
    string JSONStr = buffer.str();
    inFile.close();
    
    int InputW = ExtractIntFromJSON(JSONStr, "input_width");
    int InputH = ExtractIntFromJSON(JSONStr, "input_height");
    int InputC = ExtractIntFromJSON(JSONStr, "input_channels");
    int OutSize = ExtractIntFromJSON(JSONStr, "output_size");
    double LR = ExtractDoubleFromJSON(JSONStr, "learning_rate");
    double GradClip = ExtractDoubleFromJSON(JSONStr, "gradient_clip");
    
    // Store input dimensions
    FInputWidth = InputW;
    FInputHeight = InputH;
    FInputChannels = InputC;
    OutputSize = OutSize;
    LearningRate = LR;
    GradientClip = GradClip;
    
    int NumConvLayers = ConvLayers.size();
    int NumPoolLayers = PoolLayers.size();
    int NumFCLayers = FullyConnectedLayers.size();
    
    // Load conv layers
    for (int i = 0; i < NumConvLayers; i++) {
        for (size_t f = 0; f < ConvLayers[i].Filters.size(); f++) {
            double bias = ExtractDoubleFromJSONArray(JSONStr, "conv_layers", i, "bias");
            ConvLayers[i].Filters[f].Bias = bias;
        }
    }
    
    // Load FC layers
    for (int i = 0; i < NumFCLayers; i++) {
        for (size_t j = 0; j < FullyConnectedLayers[i].Neurons.size(); j++) {
            double bias = ExtractDoubleFromJSONArray(JSONStr, "fc_layers", i, "bias");
            FullyConnectedLayers[i].Neurons[j].Bias = bias;
        }
    }
    
    // Load output layer
    for (size_t j = 0; j < OutputLayer.Neurons.size(); j++) {
        LoadWeights1DFromJSONOutputLayer(JSONStr, j, OutputLayer.Neurons[j].Weights);
    }
    
    cout << "Model loaded from: " << Filename << endl;
}

void TCNNFacade::InitializeConvLayer(TConvLayer& Layer, int NumFilters, int InputChannels,
                                      int KernelSize, int Stride, int Padding) {
    Layer.Filters.resize(NumFilters);
    Layer.Stride = Stride;
    Layer.Padding = Padding;
    Layer.KernelSize = KernelSize;
    Layer.InputChannels = InputChannels;

    random_device rd;
    mt19937 gen(rd());
    double stddev = sqrt(2.0 / (InputChannels * KernelSize * KernelSize));
    normal_distribution<double> dist(0.0, stddev);

    for (int f = 0; f < NumFilters; f++) {
        Layer.Filters[f].Weights.resize(InputChannels);
        Layer.Filters[f].WeightsM.resize(InputChannels);
        Layer.Filters[f].WeightsV.resize(InputChannels);
        Layer.Filters[f].WeightGrads.resize(InputChannels);

        for (int c = 0; c < InputChannels; c++) {
            Layer.Filters[f].Weights[c].resize(KernelSize);
            Layer.Filters[f].WeightsM[c].resize(KernelSize);
            Layer.Filters[f].WeightsV[c].resize(KernelSize);
            Layer.Filters[f].WeightGrads[c].resize(KernelSize);

            for (int h = 0; h < KernelSize; h++) {
                Layer.Filters[f].Weights[c][h].resize(KernelSize);
                Layer.Filters[f].WeightsM[c][h].resize(KernelSize, 0.0);
                Layer.Filters[f].WeightsV[c][h].resize(KernelSize, 0.0);
                Layer.Filters[f].WeightGrads[c][h].resize(KernelSize, 0.0);

                for (int w = 0; w < KernelSize; w++) {
                    Layer.Filters[f].Weights[c][h][w] = dist(gen);
                }
            }
        }

        Layer.Filters[f].Bias = 0.0;
        Layer.Filters[f].BiasGrad = 0.0;
        Layer.Filters[f].BiasM = 0.0;
        Layer.Filters[f].BiasV = 0.0;
        
        // Allocate GPU buffers
        cl_int err;
        size_t weightSize = InputChannels * KernelSize * KernelSize * sizeof(double);
        Layer.Filters[f].WeightsBuffer = clCreateBuffer(gOpenCLContext.context, CL_MEM_READ_WRITE, 
                                                        weightSize, nullptr, &err);
        Layer.Filters[f].WeightGradsBuffer = clCreateBuffer(gOpenCLContext.context, CL_MEM_READ_WRITE,
                                                           weightSize, nullptr, &err);
    }
}

void TCNNFacade::InitializePoolLayer(TPoolingLayer& Layer, int PoolSize, int Stride) {
    Layer.PoolSize = PoolSize;
    Layer.Stride = Stride;
}

void TCNNFacade::InitializeFCLayer(TFullyConnectedLayer& Layer, int NumNeurons, int NumInputs) {
    Layer.Neurons.resize(NumNeurons);

    random_device rd;
    mt19937 gen(rd());
    double stddev = sqrt(2.0 / NumInputs);
    normal_distribution<double> dist(0.0, stddev);

    for (int n = 0; n < NumNeurons; n++) {
        Layer.Neurons[n].Weights.resize(NumInputs);
        Layer.Neurons[n].WeightsM.resize(NumInputs, 0.0);
        Layer.Neurons[n].WeightsV.resize(NumInputs, 0.0);

        for (int i = 0; i < NumInputs; i++) {
            Layer.Neurons[n].Weights[i] = dist(gen);
        }

        Layer.Neurons[n].Bias = 0.0;
        Layer.Neurons[n].BiasM = 0.0;
        Layer.Neurons[n].BiasV = 0.0;
        Layer.Neurons[n].Output = 0.0;
        Layer.Neurons[n].PreActivation = 0.0;
        Layer.Neurons[n].Error = 0.0;
        Layer.Neurons[n].DropoutMask = 1.0;
    }
}

void TCNNFacade::ConvForward(TConvLayer& Layer, const D3array& Input, int InputWidth, int InputHeight) {
    Layer.InputCache = Input;
    Layer.PaddedInput = Pad3D(Input, Layer.Padding);

    int PaddedHeight = Layer.PaddedInput[0].size();
    int PaddedWidth = Layer.PaddedInput[0][0].size();

    int OutputHeight = (PaddedHeight - Layer.KernelSize) / Layer.Stride + 1;
    int OutputWidth = (PaddedWidth - Layer.KernelSize) / Layer.Stride + 1;

    int NumFilters = Layer.Filters.size();
    Layer.OutputMaps.resize(NumFilters);
    Layer.PreActivation.resize(NumFilters);

    for (int f = 0; f < NumFilters; f++) {
        Layer.OutputMaps[f].resize(OutputHeight);
        Layer.PreActivation[f].resize(OutputHeight);

        for (int h = 0; h < OutputHeight; h++) {
            Layer.OutputMaps[f][h].resize(OutputWidth);
            Layer.PreActivation[f][h].resize(OutputWidth);

            for (int w = 0; w < OutputWidth; w++) {
                double sum = Layer.Filters[f].Bias;

                for (int c = 0; c < Layer.InputChannels; c++) {
                    for (int kh = 0; kh < Layer.KernelSize; kh++) {
                        for (int kw = 0; kw < Layer.KernelSize; kw++) {
                            int inputH = h * Layer.Stride + kh;
                            int inputW = w * Layer.Stride + kw;
                            sum += Layer.PaddedInput[c][inputH][inputW] *
                                   Layer.Filters[f].Weights[c][kh][kw];
                        }
                    }
                }

                Layer.PreActivation[f][h][w] = sum;
                Layer.OutputMaps[f][h][w] = ReLU(sum);
            }
        }
    }
}

void TCNNFacade::PoolForward(TPoolingLayer& Layer, const D3array& Input, int InputWidth, int InputHeight) {
    Layer.InputCache = Input;

    int NumChannels = Input.size();
    int OutputHeight = InputHeight / Layer.PoolSize;
    int OutputWidth = InputWidth / Layer.PoolSize;

    Layer.OutputMaps.resize(NumChannels);
    Layer.MaxIndices.resize(NumChannels);

    for (int c = 0; c < NumChannels; c++) {
        Layer.OutputMaps[c].resize(OutputHeight);
        Layer.MaxIndices[c].resize(OutputHeight);

        for (int h = 0; h < OutputHeight; h++) {
            Layer.OutputMaps[c][h].resize(OutputWidth);
            Layer.MaxIndices[c][h].resize(OutputWidth);

            for (int w = 0; w < OutputWidth; w++) {
                double maxVal = -1e308;
                int maxX = 0, maxY = 0;

                for (int ph = 0; ph < Layer.PoolSize; ph++) {
                    for (int pw = 0; pw < Layer.PoolSize; pw++) {
                        int inputH = h * Layer.PoolSize + ph;
                        int inputW = w * Layer.PoolSize + pw;

                        if (inputH < (int)Input[c].size() && inputW < (int)Input[c][0].size()) {
                            double val = Input[c][inputH][inputW];
                            if (val > maxVal) {
                                maxVal = val;
                                maxX = pw;
                                maxY = ph;
                            }
                        }
                    }
                }

                Layer.OutputMaps[c][h][w] = maxVal;
                Layer.MaxIndices[c][h][w].X = maxX;
                Layer.MaxIndices[c][h][w].Y = maxY;
            }
        }
    }
}

void TCNNFacade::FlattenFeatures(const D3array& Input, int InputWidth, int InputHeight, int InputChannels) {
    int size = InputWidth * InputHeight * InputChannels;
    FFlattenedFeatures.resize(size);

    int idx = 0;
    for (int c = 0; c < InputChannels; c++) {
        for (int h = 0; h < InputHeight; h++) {
            for (int w = 0; w < InputWidth; w++) {
                FFlattenedFeatures[idx++] = Input[c][h][w];
            }
        }
    }
}

void TCNNFacade::FCForward(TFullyConnectedLayer& Layer, const Darray& Input) {
    Layer.InputCache = Input;

    for (size_t n = 0; n < Layer.Neurons.size(); n++) {
        double sum = Layer.Neurons[n].Bias;

        for (size_t i = 0; i < Input.size(); i++) {
            sum += Input[i] * Layer.Neurons[n].Weights[i];
        }

        Layer.Neurons[n].PreActivation = sum;
        Layer.Neurons[n].Output = ReLU(sum) * Layer.Neurons[n].DropoutMask;
    }
}

void TCNNFacade::ApplyDropout(TFullyConnectedLayer& Layer) {
    if (!IsTraining || DropoutRate <= 0.0) {
        for (size_t n = 0; n < Layer.Neurons.size(); n++) {
            Layer.Neurons[n].DropoutMask = 1.0;
        }
        return;
    }

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(0.0, 1.0);
    double scale = 1.0 / (1.0 - DropoutRate);

    for (size_t n = 0; n < Layer.Neurons.size(); n++) {
        if (dist(gen) < DropoutRate) {
            Layer.Neurons[n].DropoutMask = 0.0;
        } else {
            Layer.Neurons[n].DropoutMask = scale;
        }
    }
}

D3array TCNNFacade::ConvBackward(TConvLayer& Layer, const D3array& Grad) {
    int NumFilters = Layer.Filters.size();
    int OutputHeight = Grad[0].size();
    int OutputWidth = Grad[0][0].size();
    int PaddedHeight = Layer.PaddedInput[0].size();
    int PaddedWidth = Layer.PaddedInput[0][0].size();

    D3array InputGrad(Layer.InputChannels,
                      D2array(PaddedHeight, Darray(PaddedWidth, 0.0)));

    for (int f = 0; f < NumFilters; f++) {
        for (int c = 0; c < Layer.InputChannels; c++) {
            for (int kh = 0; kh < Layer.KernelSize; kh++) {
                for (int kw = 0; kw < Layer.KernelSize; kw++) {
                    Layer.Filters[f].WeightGrads[c][kh][kw] = 0.0;
                }
            }
        }
        Layer.Filters[f].BiasGrad = 0.0;

        for (int h = 0; h < OutputHeight; h++) {
            for (int w = 0; w < OutputWidth; w++) {
                double dL_dOut = Grad[f][h][w];
                double dL_dZ = dL_dOut * ReLUDerivative(Layer.PreActivation[f][h][w]);

                Layer.Filters[f].BiasGrad += dL_dZ;

                for (int c = 0; c < Layer.InputChannels; c++) {
                    for (int kh = 0; kh < Layer.KernelSize; kh++) {
                        for (int kw = 0; kw < Layer.KernelSize; kw++) {
                            int inputH = h * Layer.Stride + kh;
                            int inputW = w * Layer.Stride + kw;

                            Layer.Filters[f].WeightGrads[c][kh][kw] +=
                                dL_dZ * Layer.PaddedInput[c][inputH][inputW];

                            InputGrad[c][inputH][inputW] +=
                                dL_dZ * Layer.Filters[f].Weights[c][kh][kw];
                        }
                    }
                }
            }
        }

        for (int c = 0; c < Layer.InputChannels; c++) {
            for (int kh = 0; kh < Layer.KernelSize; kh++) {
                for (int kw = 0; kw < Layer.KernelSize; kw++) {
                    Layer.Filters[f].WeightGrads[c][kh][kw] =
                        ClipGrad(Layer.Filters[f].WeightGrads[c][kh][kw]);
                }
            }
        }
        Layer.Filters[f].BiasGrad = ClipGrad(Layer.Filters[f].BiasGrad);
    }

    if (Layer.Padding == 0) {
        return InputGrad;
    } else {
        int OrigHeight = Layer.InputCache[0].size();
        int OrigWidth = Layer.InputCache[0][0].size();
        D3array UnpaddedGrad(Layer.InputChannels,
                             D2array(OrigHeight, Darray(OrigWidth, 0.0)));

        for (int c = 0; c < Layer.InputChannels; c++) {
            for (int h = 0; h < OrigHeight; h++) {
                for (int w = 0; w < OrigWidth; w++) {
                    UnpaddedGrad[c][h][w] = InputGrad[c][h + Layer.Padding][w + Layer.Padding];
                }
            }
        }
        return UnpaddedGrad;
    }
}

D3array TCNNFacade::PoolBackward(TPoolingLayer& Layer, const D3array& Grad) {
    int NumChannels = Layer.InputCache.size();
    int InputHeight = Layer.InputCache[0].size();
    int InputWidth = Layer.InputCache[0][0].size();

    D3array InputGrad(NumChannels, D2array(InputHeight, Darray(InputWidth, 0.0)));

    int OutputHeight = Grad[0].size();
    int OutputWidth = Grad[0][0].size();

    for (int c = 0; c < NumChannels; c++) {
        for (int h = 0; h < OutputHeight; h++) {
            for (int w = 0; w < OutputWidth; w++) {
                int maxX = Layer.MaxIndices[c][h][w].X;
                int maxY = Layer.MaxIndices[c][h][w].Y;

                int inputH = h * Layer.PoolSize + maxY;
                int inputW = w * Layer.PoolSize + maxX;

                if (inputH < InputHeight && inputW < InputWidth) {
                    InputGrad[c][inputH][inputW] += Grad[c][h][w];
                }
            }
        }
    }

    return InputGrad;
}

Darray TCNNFacade::FCBackward(TFullyConnectedLayer& Layer, const Darray& Grad, bool IsOutputLayer) {
    int NumNeurons = Layer.Neurons.size();
    int InputSize = Layer.InputCache.size();

    Darray InputGrad(InputSize, 0.0);

    for (int n = 0; n < NumNeurons; n++) {
        double dL_dOut = Grad[n];

        double dL_dZ;
        if (IsOutputLayer) {
            dL_dZ = dL_dOut;
        } else {
            dL_dZ = dL_dOut * ReLUDerivative(Layer.Neurons[n].PreActivation) *
                    Layer.Neurons[n].DropoutMask;
        }

        Layer.Neurons[n].Error = dL_dZ;

        for (int i = 0; i < InputSize; i++) {
            InputGrad[i] += dL_dZ * Layer.Neurons[n].Weights[i];
        }
    }

    return InputGrad;
}

D3array TCNNFacade::UnflattenGradient(const Darray& Grad) {
    D3array Result(LastConvChannels,
                   D2array(LastConvHeight, Darray(LastConvWidth, 0.0)));

    int idx = 0;
    for (int c = 0; c < LastConvChannels; c++) {
        for (int h = 0; h < LastConvHeight; h++) {
            for (int w = 0; w < LastConvWidth; w++) {
                if (idx < (int)Grad.size()) {
                    Result[c][h][w] = Grad[idx++];
                }
            }
        }
    }

    return Result;
}

void TCNNFacade::UpdateWeights() {
    AdamT++;

    double beta1_t = pow(Beta1, AdamT);
    double beta2_t = pow(Beta2, AdamT);

    // Update convolutional layers
    for (size_t layer = 0; layer < ConvLayers.size(); layer++) {
        for (size_t f = 0; f < ConvLayers[layer].Filters.size(); f++) {
            for (int c = 0; c < ConvLayers[layer].InputChannels; c++) {
                for (int kh = 0; kh < ConvLayers[layer].KernelSize; kh++) {
                    for (int kw = 0; kw < ConvLayers[layer].KernelSize; kw++) {
                        double grad = ConvLayers[layer].Filters[f].WeightGrads[c][kh][kw];

                        ConvLayers[layer].Filters[f].WeightsM[c][kh][kw] =
                            Beta1 * ConvLayers[layer].Filters[f].WeightsM[c][kh][kw] +
                            (1.0 - Beta1) * grad;

                        ConvLayers[layer].Filters[f].WeightsV[c][kh][kw] =
                            Beta2 * ConvLayers[layer].Filters[f].WeightsV[c][kh][kw] +
                            (1.0 - Beta2) * grad * grad;

                        double m_hat = ConvLayers[layer].Filters[f].WeightsM[c][kh][kw] / (1.0 - beta1_t);
                        double v_hat = ConvLayers[layer].Filters[f].WeightsV[c][kh][kw] / (1.0 - beta2_t);

                        ConvLayers[layer].Filters[f].Weights[c][kh][kw] -=
                            LearningRate * m_hat / (sqrt(v_hat) + EPSILON);
                    }
                }
            }

            double grad = ConvLayers[layer].Filters[f].BiasGrad;

            ConvLayers[layer].Filters[f].BiasM =
                Beta1 * ConvLayers[layer].Filters[f].BiasM + (1.0 - Beta1) * grad;

            ConvLayers[layer].Filters[f].BiasV =
                Beta2 * ConvLayers[layer].Filters[f].BiasV + (1.0 - Beta2) * grad * grad;

            double m_hat = ConvLayers[layer].Filters[f].BiasM / (1.0 - beta1_t);
            double v_hat = ConvLayers[layer].Filters[f].BiasV / (1.0 - beta2_t);

            ConvLayers[layer].Filters[f].Bias -=
                LearningRate * m_hat / (sqrt(v_hat) + EPSILON);
        }
    }

    // Update fully connected layers
    for (size_t layer = 0; layer < FullyConnectedLayers.size(); layer++) {
        for (size_t n = 0; n < FullyConnectedLayers[layer].Neurons.size(); n++) {
            double error = FullyConnectedLayers[layer].Neurons[n].Error;

            for (size_t i = 0; i < FullyConnectedLayers[layer].Neurons[n].Weights.size(); i++) {
                double grad = error * FullyConnectedLayers[layer].InputCache[i];
                grad = ClipGrad(grad);

                FullyConnectedLayers[layer].Neurons[n].WeightsM[i] =
                    Beta1 * FullyConnectedLayers[layer].Neurons[n].WeightsM[i] +
                    (1.0 - Beta1) * grad;

                FullyConnectedLayers[layer].Neurons[n].WeightsV[i] =
                    Beta2 * FullyConnectedLayers[layer].Neurons[n].WeightsV[i] +
                    (1.0 - Beta2) * grad * grad;

                double m_hat = FullyConnectedLayers[layer].Neurons[n].WeightsM[i] / (1.0 - beta1_t);
                double v_hat = FullyConnectedLayers[layer].Neurons[n].WeightsV[i] / (1.0 - beta2_t);

                FullyConnectedLayers[layer].Neurons[n].Weights[i] -=
                    LearningRate * m_hat / (sqrt(v_hat) + EPSILON);
            }

            double grad = error;
            grad = ClipGrad(grad);

            FullyConnectedLayers[layer].Neurons[n].BiasM =
                Beta1 * FullyConnectedLayers[layer].Neurons[n].BiasM + (1.0 - Beta1) * grad;

            FullyConnectedLayers[layer].Neurons[n].BiasV =
                Beta2 * FullyConnectedLayers[layer].Neurons[n].BiasV + (1.0 - Beta2) * grad * grad;

            double m_hat = FullyConnectedLayers[layer].Neurons[n].BiasM / (1.0 - beta1_t);
            double v_hat = FullyConnectedLayers[layer].Neurons[n].BiasV / (1.0 - beta2_t);

            FullyConnectedLayers[layer].Neurons[n].Bias -=
                LearningRate * m_hat / (sqrt(v_hat) + EPSILON);
        }
    }

    // Update output layer
    for (size_t n = 0; n < OutputLayer.Neurons.size(); n++) {
        double error = OutputLayer.Neurons[n].Error;

        for (size_t i = 0; i < OutputLayer.Neurons[n].Weights.size(); i++) {
            double grad = error * OutputLayer.InputCache[i];
            grad = ClipGrad(grad);

            OutputLayer.Neurons[n].WeightsM[i] =
                Beta1 * OutputLayer.Neurons[n].WeightsM[i] + (1.0 - Beta1) * grad;

            OutputLayer.Neurons[n].WeightsV[i] =
                Beta2 * OutputLayer.Neurons[n].WeightsV[i] + (1.0 - Beta2) * grad * grad;

            double m_hat = OutputLayer.Neurons[n].WeightsM[i] / (1.0 - beta1_t);
            double v_hat = OutputLayer.Neurons[n].WeightsV[i] / (1.0 - beta2_t);

            OutputLayer.Neurons[n].Weights[i] -=
                LearningRate * m_hat / (sqrt(v_hat) + EPSILON);
        }

        double grad = error;
        grad = ClipGrad(grad);

        OutputLayer.Neurons[n].BiasM =
            Beta1 * OutputLayer.Neurons[n].BiasM + (1.0 - Beta1) * grad;

        OutputLayer.Neurons[n].BiasV =
            Beta2 * OutputLayer.Neurons[n].BiasV + (1.0 - Beta2) * grad * grad;

        double m_hat = OutputLayer.Neurons[n].BiasM / (1.0 - beta1_t);
        double v_hat = OutputLayer.Neurons[n].BiasV / (1.0 - beta2_t);

        OutputLayer.Neurons[n].Bias -=
            LearningRate * m_hat / (sqrt(v_hat) + EPSILON);
    }
}

// Input dimension getters
int TCNNFacade::GetInputWidth() { return FInputWidth; }
int TCNNFacade::GetInputHeight() { return FInputHeight; }
int TCNNFacade::GetInputChannels() { return FInputChannels; }

// Training parameter getters
int TCNNFacade::GetOutputSize() { return this->OutputSize; }
double TCNNFacade::GetLearningRate() { return LearningRate; }
double TCNNFacade::GetGradientClip() { return GradientClip; }

string TCNNFacade::GetActivationTypeString() {
    switch (ActivationType) {
        case 0: return "sigmoid";
        case 1: return "tanh";
        case 2: return "relu";
        case 3: return "linear";
        default: return "unknown";
    }
}

string TCNNFacade::GetOutputActivationTypeString() {
    switch (OutputActivationType) {
        case 0: return "sigmoid";
        case 1: return "tanh";
        case 2: return "relu";
        case 3: return "linear";
        default: return "unknown";
    }
}

int TCNNFacade::GetLossType() { return LossType; }

// Feature Map Access
D2array TCNNFacade::GetFeatureMap(int LayerIdx, int FilterIdx) {
    if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return D2array();
    if (FilterIdx < 0 || FilterIdx >= (int)ConvLayers[LayerIdx].OutputMaps.size()) return D2array();
    return ConvLayers[LayerIdx].OutputMaps[FilterIdx];
}

void TCNNFacade::SetFeatureMap(int LayerIdx, int FilterIdx, const D2array& Map) {
    if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return;
    if (FilterIdx < 0 || FilterIdx >= (int)ConvLayers[LayerIdx].OutputMaps.size()) return;
    ConvLayers[LayerIdx].OutputMaps[FilterIdx] = Map;
}

// Pre-Activation Access
D2array TCNNFacade::GetPreActivation(int LayerIdx, int FilterIdx) {
    if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return D2array();
    if (FilterIdx < 0 || FilterIdx >= (int)ConvLayers[LayerIdx].PreActivation.size()) return D2array();
    return ConvLayers[LayerIdx].PreActivation[FilterIdx];
}

void TCNNFacade::SetPreActivation(int LayerIdx, int FilterIdx, const D2array& Map) {
    if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return;
    if (FilterIdx < 0 || FilterIdx >= (int)ConvLayers[LayerIdx].PreActivation.size()) return;
    ConvLayers[LayerIdx].PreActivation[FilterIdx] = Map;
}

// Kernel/Filter Access
D2array TCNNFacade::GetKernel(int LayerIdx, int FilterIdx, int ChannelIdx) {
    if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return D2array();
    if (FilterIdx < 0 || FilterIdx >= (int)ConvLayers[LayerIdx].Filters.size()) return D2array();
    if (ChannelIdx < 0 || ChannelIdx >= (int)ConvLayers[LayerIdx].Filters[FilterIdx].Weights.size()) return D2array();
    return ConvLayers[LayerIdx].Filters[FilterIdx].Weights[ChannelIdx];
}

void TCNNFacade::SetKernel(int LayerIdx, int FilterIdx, int ChannelIdx, const D2array& KernelArray) {
    if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return;
    if (FilterIdx < 0 || FilterIdx >= (int)ConvLayers[LayerIdx].Filters.size()) return;
    if (ChannelIdx < 0 || ChannelIdx >= (int)ConvLayers[LayerIdx].Filters[FilterIdx].Weights.size()) return;
    ConvLayers[LayerIdx].Filters[FilterIdx].Weights[ChannelIdx] = KernelArray;
}

double TCNNFacade::GetBias(int LayerIdx, int FilterIdx) {
    if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return 0.0;
    if (FilterIdx < 0 || FilterIdx >= (int)ConvLayers[LayerIdx].Filters.size()) return 0.0;
    return ConvLayers[LayerIdx].Filters[FilterIdx].Bias;
}

void TCNNFacade::SetBias(int LayerIdx, int FilterIdx, double Value) {
    if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return;
    if (FilterIdx < 0 || FilterIdx >= (int)ConvLayers[LayerIdx].Filters.size()) return;
    ConvLayers[LayerIdx].Filters[FilterIdx].Bias = Value;
}

// Batch Activations
D4array TCNNFacade::GetBatchActivations(int LayerIdx) {
    if (LayerIdx < 0 || LayerIdx >= (int)FBatchActivations.size()) return D4array();
    return FBatchActivations[LayerIdx];
}

void TCNNFacade::SetBatchActivations(int LayerIdx, const D4array& BatchTensor) {
    if (LayerIdx < 0) return;
    if (LayerIdx >= (int)FBatchActivations.size()) FBatchActivations.resize(LayerIdx + 1);
    FBatchActivations[LayerIdx] = BatchTensor;
}

// Pooling Indices
D2array TCNNFacade::GetPoolingIndices(int LayerIdx, int FilterIdx) {
    if (LayerIdx < 0 || LayerIdx >= (int)PoolLayers.size()) return D2array();
    if (FilterIdx < 0 || FilterIdx >= (int)PoolLayers[LayerIdx].MaxIndices.size()) return D2array();

    int height = PoolLayers[LayerIdx].MaxIndices[FilterIdx].size();
    int width = PoolLayers[LayerIdx].MaxIndices[FilterIdx][0].size();
    D2array Result(height, Darray(width * 2));

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            Result[h][w * 2] = PoolLayers[LayerIdx].MaxIndices[FilterIdx][h][w].X;
            Result[h][w * 2 + 1] = PoolLayers[LayerIdx].MaxIndices[FilterIdx][h][w].Y;
        }
    }
    return Result;
}

// Dropout Mask
Darray TCNNFacade::GetDropoutMask(int LayerIdx) {
    if (LayerIdx < 0 || LayerIdx >= (int)FullyConnectedLayers.size()) return Darray();
    Darray Result(FullyConnectedLayers[LayerIdx].Neurons.size());
    for (size_t i = 0; i < FullyConnectedLayers[LayerIdx].Neurons.size(); i++) {
        Result[i] = FullyConnectedLayers[LayerIdx].Neurons[i].DropoutMask;
    }
    return Result;
}

void TCNNFacade::SetDropoutMask(int LayerIdx, const Darray& Mask) {
    if (LayerIdx < 0 || LayerIdx >= (int)FullyConnectedLayers.size()) return;
    for (size_t i = 0; i < Mask.size() && i < FullyConnectedLayers[LayerIdx].Neurons.size(); i++) {
        FullyConnectedLayers[LayerIdx].Neurons[i].DropoutMask = Mask[i];
    }
}

// Gradients
D2array TCNNFacade::GetFilterGradient(int LayerIdx, int FilterIdx, int ChannelIdx) {
    if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return D2array();
    if (FilterIdx < 0 || FilterIdx >= (int)ConvLayers[LayerIdx].Filters.size()) return D2array();
    if (ChannelIdx < 0 || ChannelIdx >= (int)ConvLayers[LayerIdx].Filters[FilterIdx].WeightGrads.size()) return D2array();
    return ConvLayers[LayerIdx].Filters[FilterIdx].WeightGrads[ChannelIdx];
}

double TCNNFacade::GetBiasGradient(int LayerIdx, int FilterIdx) {
    if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return 0.0;
    if (FilterIdx < 0 || FilterIdx >= (int)ConvLayers[LayerIdx].Filters.size()) return 0.0;
    return ConvLayers[LayerIdx].Filters[FilterIdx].BiasGrad;
}

double TCNNFacade::GetActivationGradient(int LayerIdx, int FilterIdx, int Y, int X) {
    return 0.0;
}

D3array TCNNFacade::GetOptimizerState(int LayerIdx, int FilterIdx, const string& Param) {
    if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return D3array();
    if (FilterIdx < 0 || FilterIdx >= (int)ConvLayers[LayerIdx].Filters.size()) return D3array();

    if (Param == "m" || Param == "M" || Param == "momentum") {
        return ConvLayers[LayerIdx].Filters[FilterIdx].WeightsM;
    } else if (Param == "v" || Param == "V" || Param == "variance") {
        return ConvLayers[LayerIdx].Filters[FilterIdx].WeightsV;
    } else {
        return D3array();
    }
}

// Flattened Features
Darray TCNNFacade::GetFlattenedFeatures() { return FFlattenedFeatures; }
void TCNNFacade::SetFlattenedFeatures(const Darray& Vector) { FFlattenedFeatures = Vector; }

// Output Layer
Darray TCNNFacade::GetLogits() {
    Darray Result(OutputLayer.Neurons.size());
    for (size_t i = 0; i < OutputLayer.Neurons.size(); i++) {
        Result[i] = OutputLayer.Neurons[i].PreActivation;
    }
    return Result;
}

Darray TCNNFacade::GetSoftmax() {
    Darray Result(OutputLayer.Neurons.size());
    for (size_t i = 0; i < OutputLayer.Neurons.size(); i++) {
        Result[i] = OutputLayer.Neurons[i].Output;
    }
    return Result;
}

// Layer Config
TLayerConfig TCNNFacade::GetLayerConfig(int LayerIdx) {
    TLayerConfig config;
    int totalLayers = ConvLayers.size() + PoolLayers.size() + FullyConnectedLayers.size() + 1;

    if (LayerIdx < 0 || LayerIdx >= totalLayers) {
        config.LayerType = "invalid";
        return config;
    }

    int currentIdx = 0;

    for (size_t i = 0; i < ConvLayers.size(); i++) {
        if (currentIdx == LayerIdx) {
            config.LayerType = "conv";
            config.FilterCount = ConvLayers[i].Filters.size();
            config.KernelSize = ConvLayers[i].KernelSize;
            config.Stride = ConvLayers[i].Stride;
            config.Padding = ConvLayers[i].Padding;
            config.InputChannels = ConvLayers[i].InputChannels;
            if (!ConvLayers[i].OutputMaps.empty() && !ConvLayers[i].OutputMaps[0].empty()) {
                config.OutputHeight = ConvLayers[i].OutputMaps[0].size();
                config.OutputWidth = ConvLayers[i].OutputMaps[0][0].size();
            } else {
                config.OutputHeight = 0;
                config.OutputWidth = 0;
            }
            config.PoolSize = 0;
            config.NeuronCount = 0;
            config.InputSize = 0;
            return config;
        }
        currentIdx++;

        if (i < PoolLayers.size()) {
            if (currentIdx == LayerIdx) {
                config.LayerType = "pool";
                config.PoolSize = PoolLayers[i].PoolSize;
                config.Stride = PoolLayers[i].Stride;
                if (!PoolLayers[i].OutputMaps.empty() && !PoolLayers[i].OutputMaps[0].empty()) {
                    config.OutputHeight = PoolLayers[i].OutputMaps[0].size();
                    config.OutputWidth = PoolLayers[i].OutputMaps[0][0].size();
                } else {
                    config.OutputHeight = 0;
                    config.OutputWidth = 0;
                }
                config.FilterCount = 0;
                config.KernelSize = 0;
                config.Padding = 0;
                config.InputChannels = 0;
                config.NeuronCount = 0;
                config.InputSize = 0;
                return config;
            }
            currentIdx++;
        }
    }

    for (size_t i = 0; i < FullyConnectedLayers.size(); i++) {
        if (currentIdx == LayerIdx) {
            config.LayerType = "fc";
            config.NeuronCount = FullyConnectedLayers[i].Neurons.size();
            if (!FullyConnectedLayers[i].Neurons.empty()) {
                config.InputSize = FullyConnectedLayers[i].Neurons[0].Weights.size();
            } else {
                config.InputSize = 0;
            }
            return config;
        }
        currentIdx++;
    }

    return config;
}

int TCNNFacade::GetNumLayers() {
    return ConvLayers.size() + PoolLayers.size() + FullyConnectedLayers.size() + 1;
}

int TCNNFacade::GetNumConvLayers() { return ConvLayers.size(); }
int TCNNFacade::GetNumFCLayers() { return FullyConnectedLayers.size(); }

int TCNNFacade::GetNumFilters(int LayerIdx) {
    if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return 0;
    return ConvLayers[LayerIdx].Filters.size();
}

// Placeholder methods for advanced features
D2array TCNNFacade::GetSaliencyMap(int LayerIdx, int FilterIdx, int InputIdx) { return D2array(); }
D3array TCNNFacade::GetDeconv(int LayerIdx, int FilterIdx, bool UpToInput) { return D3array(); }
void TCNNFacade::AddFilter(int LayerIdx, const D3array& Params) {}
void TCNNFacade::RemoveFilter(int LayerIdx, int FilterIdx) {}
void TCNNFacade::AddConvLayer(int Position, int NumFilters, int KernelSize, int Stride, int Padding) {}
void TCNNFacade::RemoveLayer(int LayerIdx) {}
TLayerStats TCNNFacade::GetLayerStats(int LayerIdx) { return TLayerStats(); }
Darray TCNNFacade::GetActivationHistogram(int LayerIdx, int NumBins) { return Darray(); }
Darray TCNNFacade::GetWeightHistogram(int LayerIdx, int NumBins) { return Darray(); }
TReceptiveField TCNNFacade::GetReceptiveField(int LayerIdx, int FeatureIdx, int Y, int X) { return TReceptiveField(); }
TBatchNormParams TCNNFacade::GetBatchNormParams(int LayerIdx) { return TBatchNormParams(); }
void TCNNFacade::SetBatchNormParams(int LayerIdx, const TBatchNormParams& Params) {}
void TCNNFacade::SetFilterAttribute(int LayerIdx, int FilterIdx, const string& Key, const string& Value) {}
string TCNNFacade::GetFilterAttribute(int LayerIdx, int FilterIdx, const string& Key) { return ""; }
void TCNNFacade::SetTrainingMode(bool Training) { IsTraining = Training; }
bool TCNNFacade::GetTrainingMode() { return IsTraining; }

void TCNNFacade::SaveModel(const string& Filename) {
    SaveModelToJSON(Filename);
}

void TCNNFacade::LoadModel(const string& Filename) {
    LoadModelFromJSON(Filename);
}

/* ========== ONNX Export/Import ========== */

void TCNNFacade::ExportToONNX(const string& Filename) {
    ofstream file(Filename, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file for writing: " + Filename);
    }
    
    // Write ONNX magic header
    const char magic[] = "ONNX";
    file.write(magic, 4);
    
    // Write version
    int version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(int));
    
    // Write model metadata
    int inputW = FInputWidth;
    int inputH = FInputHeight;
    int inputC = FInputChannels;
    int outSize = OutputLayer.Neurons.size();
    file.write(reinterpret_cast<const char*>(&inputW), sizeof(int));
    file.write(reinterpret_cast<const char*>(&inputH), sizeof(int));
    file.write(reinterpret_cast<const char*>(&inputC), sizeof(int));
    file.write(reinterpret_cast<const char*>(&outSize), sizeof(int));
    
    // Write batch norm flag (facade version doesn't have full batch norm support, write 0)
    int useBN = 0;
    file.write(reinterpret_cast<const char*>(&useBN), sizeof(int));
    
    // Write conv layer count and metadata
    int numConvLayers = ConvLayers.size();
    file.write(reinterpret_cast<const char*>(&numConvLayers), sizeof(int));
    
    for (int i = 0; i < numConvLayers; i++) {
        int numFilters = ConvLayers[i].Filters.size();
        int kernelSize = ConvLayers[i].KernelSize;
        int poolSize = (i < (int)PoolLayers.size()) ? PoolLayers[i].PoolSize : 2;
        file.write(reinterpret_cast<const char*>(&numFilters), sizeof(int));
        file.write(reinterpret_cast<const char*>(&kernelSize), sizeof(int));
        file.write(reinterpret_cast<const char*>(&poolSize), sizeof(int));
    }
    
    // Write FC layer count and sizes
    int numFCLayers = FullyConnectedLayers.size();
    file.write(reinterpret_cast<const char*>(&numFCLayers), sizeof(int));
    for (int i = 0; i < numFCLayers; i++) {
        int fcSize = FullyConnectedLayers[i].Neurons.size();
        file.write(reinterpret_cast<const char*>(&fcSize), sizeof(int));
    }
    
    // Write conv layer weights
    for (size_t i = 0; i < ConvLayers.size(); i++) {
        int numFilters = ConvLayers[i].Filters.size();
        file.write(reinterpret_cast<const char*>(&numFilters), sizeof(int));
        
        for (int f = 0; f < numFilters; f++) {
            TConvFilter& filter = ConvLayers[i].Filters[f];
            // Write filter weights dimensions (3D: Channels x H x W)
            int d0 = filter.Weights.size();
            int d1 = d0 > 0 ? filter.Weights[0].size() : 0;
            int d2 = d1 > 0 ? filter.Weights[0][0].size() : 0;
            
            file.write(reinterpret_cast<const char*>(&d0), sizeof(int));
            file.write(reinterpret_cast<const char*>(&d1), sizeof(int));
            file.write(reinterpret_cast<const char*>(&d2), sizeof(int));
            
            for (int a = 0; a < d0; a++) {
                for (int b = 0; b < d1; b++) {
                    for (int c = 0; c < d2; c++) {
                        file.write(reinterpret_cast<const char*>(&filter.Weights[a][b][c]), sizeof(double));
                    }
                }
            }
            
            // Write bias
            file.write(reinterpret_cast<const char*>(&filter.Bias), sizeof(double));
        }
    }
    
    // Write FC layer weights
    for (size_t i = 0; i < FullyConnectedLayers.size(); i++) {
        int numNeurons = FullyConnectedLayers[i].Neurons.size();
        int inputSize = numNeurons > 0 ? FullyConnectedLayers[i].Neurons[0].Weights.size() : 0;
        
        file.write(reinterpret_cast<const char*>(&numNeurons), sizeof(int));
        file.write(reinterpret_cast<const char*>(&inputSize), sizeof(int));
        
        for (int n = 0; n < numNeurons; n++) {
            for (int w = 0; w < inputSize; w++) {
                file.write(reinterpret_cast<const char*>(&FullyConnectedLayers[i].Neurons[n].Weights[w]), sizeof(double));
            }
        }
        
        for (int n = 0; n < numNeurons; n++) {
            file.write(reinterpret_cast<const char*>(&FullyConnectedLayers[i].Neurons[n].Bias), sizeof(double));
        }
    }
    
    // Write output layer
    int outNeurons = OutputLayer.Neurons.size();
    int outInputSize = outNeurons > 0 ? OutputLayer.Neurons[0].Weights.size() : 0;
    file.write(reinterpret_cast<const char*>(&outNeurons), sizeof(int));
    file.write(reinterpret_cast<const char*>(&outInputSize), sizeof(int));
    
    for (int n = 0; n < outNeurons; n++) {
        for (int w = 0; w < outInputSize; w++) {
            file.write(reinterpret_cast<const char*>(&OutputLayer.Neurons[n].Weights[w]), sizeof(double));
        }
    }
    
    for (int n = 0; n < outNeurons; n++) {
        file.write(reinterpret_cast<const char*>(&OutputLayer.Neurons[n].Bias), sizeof(double));
    }
    
    file.close();
    cout << "Model exported to ONNX: " << Filename << endl;
}

TCNNFacade* TCNNFacade::ImportFromONNX(const string& Filename) {
    ifstream file(Filename, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file for reading: " + Filename);
    }
    
    // Read and verify magic header
    char magic[5] = {0};
    file.read(magic, 4);
    if (string(magic) != "ONNX") {
        throw runtime_error("Invalid ONNX file format");
    }
    
    int version;
    file.read(reinterpret_cast<char*>(&version), sizeof(int));
    
    // Read model metadata
    int inputW, inputH, inputC, outputSize, useBN;
    file.read(reinterpret_cast<char*>(&inputW), sizeof(int));
    file.read(reinterpret_cast<char*>(&inputH), sizeof(int));
    file.read(reinterpret_cast<char*>(&inputC), sizeof(int));
    file.read(reinterpret_cast<char*>(&outputSize), sizeof(int));
    file.read(reinterpret_cast<char*>(&useBN), sizeof(int));
    
    // Read conv layer metadata
    int numConvLayers;
    file.read(reinterpret_cast<char*>(&numConvLayers), sizeof(int));
    
    vector<int> convFilters(numConvLayers);
    vector<int> kernelSizes(numConvLayers);
    vector<int> poolSizes(numConvLayers);
    
    for (int i = 0; i < numConvLayers; i++) {
        file.read(reinterpret_cast<char*>(&convFilters[i]), sizeof(int));
        file.read(reinterpret_cast<char*>(&kernelSizes[i]), sizeof(int));
        file.read(reinterpret_cast<char*>(&poolSizes[i]), sizeof(int));
    }
    
    // Read FC layer sizes
    int numFCLayers;
    file.read(reinterpret_cast<char*>(&numFCLayers), sizeof(int));
    vector<int> fcSizes(numFCLayers);
    for (int i = 0; i < numFCLayers; i++) {
        file.read(reinterpret_cast<char*>(&fcSizes[i]), sizeof(int));
    }
    
    // Create CNN with loaded architecture
    TCNNFacade* cnn = new TCNNFacade(inputW, inputH, inputC,
                                      convFilters, kernelSizes, poolSizes,
                                      fcSizes, outputSize, 0.001, 0.25);
    
    // Read conv layer weights
    for (int i = 0; i < numConvLayers; i++) {
        int numFilters;
        file.read(reinterpret_cast<char*>(&numFilters), sizeof(int));
        
        for (int f = 0; f < numFilters; f++) {
            int d0, d1, d2;
            file.read(reinterpret_cast<char*>(&d0), sizeof(int));
            file.read(reinterpret_cast<char*>(&d1), sizeof(int));
            file.read(reinterpret_cast<char*>(&d2), sizeof(int));
            
            cnn->ConvLayers[i].Filters[f].Weights.resize(d0);
            for (int a = 0; a < d0; a++) {
                cnn->ConvLayers[i].Filters[f].Weights[a].resize(d1);
                for (int b = 0; b < d1; b++) {
                    cnn->ConvLayers[i].Filters[f].Weights[a][b].resize(d2);
                    for (int c = 0; c < d2; c++) {
                        file.read(reinterpret_cast<char*>(&cnn->ConvLayers[i].Filters[f].Weights[a][b][c]), sizeof(double));
                    }
                }
            }
            
            file.read(reinterpret_cast<char*>(&cnn->ConvLayers[i].Filters[f].Bias), sizeof(double));
        }
    }
    
    // Read FC layer weights
    for (int i = 0; i < (int)cnn->FullyConnectedLayers.size(); i++) {
        int numNeurons, inputSize;
        file.read(reinterpret_cast<char*>(&numNeurons), sizeof(int));
        file.read(reinterpret_cast<char*>(&inputSize), sizeof(int));
        
        for (int n = 0; n < numNeurons; n++) {
            cnn->FullyConnectedLayers[i].Neurons[n].Weights.resize(inputSize);
            for (int w = 0; w < inputSize; w++) {
                file.read(reinterpret_cast<char*>(&cnn->FullyConnectedLayers[i].Neurons[n].Weights[w]), sizeof(double));
            }
        }
        
        for (int n = 0; n < numNeurons; n++) {
            file.read(reinterpret_cast<char*>(&cnn->FullyConnectedLayers[i].Neurons[n].Bias), sizeof(double));
        }
    }
    
    // Read output layer
    int outNeurons, outInputSize;
    file.read(reinterpret_cast<char*>(&outNeurons), sizeof(int));
    file.read(reinterpret_cast<char*>(&outInputSize), sizeof(int));
    
    for (int n = 0; n < outNeurons; n++) {
        cnn->OutputLayer.Neurons[n].Weights.resize(outInputSize);
        for (int w = 0; w < outInputSize; w++) {
            file.read(reinterpret_cast<char*>(&cnn->OutputLayer.Neurons[n].Weights[w]), sizeof(double));
        }
    }
    
    for (int n = 0; n < outNeurons; n++) {
        file.read(reinterpret_cast<char*>(&cnn->OutputLayer.Neurons[n].Bias), sizeof(double));
    }
    
    file.close();
    cout << "Model imported from ONNX: " << Filename << endl;
    return cnn;
}

/* ========== COMMAND-LINE UTILITIES ========== */

void ParseIntArrayHelper(const string& s, vector<int>& Result) {
    Result.clear();
    stringstream ss(s);
    string token;
    while (getline(ss, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        if (!token.empty()) {
            Result.push_back(stoi(token));
        }
    }
}

void PrintUsage() {
    cout << "Facaded CNN" << endl;
    cout << endl;
    cout << "Commands:" << endl;
    cout << "  create      Create a new CNN model and save to JSON" << endl;
    cout << "  train       Train on sample data from JSON model" << endl;
    cout << "  predict     Make predictions with JSON model" << endl;
    cout << "  introspect  Examine layer internals (activations, weights, gradients)" << endl;
    cout << "  stats       Display layer statistics and histograms" << endl;
    cout << "  modify      Add/remove filters or layers dynamically" << endl;
    cout << "  analyze     Get saliency maps, deconv, receptive fields" << endl;
    cout << "  info        Display complete model architecture from JSON" << endl;
    cout << "  export-onnx Export model to ONNX format" << endl;
    cout << "  import-onnx Import model from ONNX format" << endl;
    cout << "  help        Show this help message" << endl;
    cout << endl;
    cout << "Create Options:" << endl;
    cout << "  --input-w=N            Input width (required)" << endl;
    cout << "  --input-h=N            Input height (required)" << endl;
    cout << "  --input-c=N            Input channels (required)" << endl;
    cout << "  --conv=N,N,...         Conv filters (required)" << endl;
    cout << "  --kernels=N,N,...      Kernel sizes (required)" << endl;
    cout << "  --pools=N,N,...        Pool sizes (required)" << endl;
    cout << "  --fc=N,N,...           FC layer sizes (required)" << endl;
    cout << "  --output=N             Output layer size (required)" << endl;
    cout << "  --save=FILE.json       Save model to JSON file (required)" << endl;
    cout << "  --lr=VALUE             Learning rate (default: 0.001)" << endl;
    cout << "  --dropout=VALUE        Dropout rate (default: 0.25)" << endl;
    cout << endl;
    cout << "Train Options:" << endl;
    cout << "  --model=FILE.json      Load model from JSON file (required)" << endl;
    cout << "  --epochs=N             Number of epochs (required)" << endl;
    cout << "  --save=FILE.json       Save trained model to JSON (required)" << endl;
    cout << endl;
    cout << "Predict Options:" << endl;
    cout << "  --model=FILE.json      Load model from JSON file (required)" << endl;
    cout << "  --random-input         Use random input data" << endl;
    cout << "  --mode=WHAT            basic | all (default: basic)" << endl;
    cout << endl;
    cout << "Introspect Options:" << endl;
    cout << "  --model=FILE.json      Load model from JSON file (optional)" << endl;
    cout << "  --layer=N              Layer index (0-based)" << endl;
    cout << "  --filter=N             Filter/neuron index" << endl;
    cout << "  --mode=WHAT            feature_map | pre_activation | kernel | bias" << endl;
    cout << "  --channel=N            Channel index (for kernels)" << endl;
    cout << endl;
    cout << "Stats Options:" << endl;
    cout << "  --model=FILE.json      Load model from JSON file (optional)" << endl;
    cout << "  --layer=N              Layer index" << endl;
    cout << "  --histogram=N          Number of bins for histogram (default: 50)" << endl;
    cout << "  --type=WHAT            weights | activations | both" << endl;
    cout << endl;
    cout << "Modify Options:" << endl;
    cout << "  --model=FILE.json      Load model from JSON file (optional)" << endl;
    cout << "  --action=ACTION        add_filter | remove_filter | add_layer" << endl;
    cout << "  --layer=N              Layer index for action" << endl;
    cout << "  --filter=N             Filter index (for remove_filter)" << endl;
    cout << "  --filters=N            Number of filters (for add_layer)" << endl;
    cout << "  --kernel=N             Kernel size (for add_layer)" << endl;
    cout << "  --stride=N             Stride (for add_layer, default: 1)" << endl;
    cout << "  --padding=N            Padding (for add_layer, default: 0)" << endl;
    cout << "  --save=FILE.json       Save modified model to JSON" << endl;
    cout << endl;
    cout << "Analyze Options:" << endl;
    cout << "  --model=FILE.json      Load model from JSON file (optional)" << endl;
    cout << "  --layer=N              Layer index" << endl;
    cout << "  --filter=N             Filter index" << endl;
    cout << "  --type=WHAT            saliency | deconv | receptive_field" << endl;
    cout << "  --x=N --y=N            Position for saliency/receptive field" << endl;
    cout << endl;
    cout << "Info Options:" << endl;
    cout << "  --model=FILE.json      Load model from JSON file (required)" << endl;
    cout << endl;
    cout << "Export ONNX Options:" << endl;
    cout << "  --model=FILE.json      Load model from JSON file (required)" << endl;
    cout << "  --onnx=FILE.onnx       Save to ONNX file (required)" << endl;
    cout << endl;
    cout << "Import ONNX Options:" << endl;
    cout << "  --onnx=FILE.onnx       Load from ONNX file (required)" << endl;
    cout << "  --save=FILE.json       Save to JSON file (required)" << endl;
    cout << endl;
    cout << "Examples:" << endl;
    cout << "  facaded_cnn create --input-w=28 --input-h=28 --input-c=1 --conv=8,16 --kernels=3,3 --pools=2,2 --fc=64 --output=10 --save=model.json" << endl;
    cout << "  facaded_cnn train --model=model.json --epochs=50 --save=model_trained.json" << endl;
    cout << "  facaded_cnn predict --model=model.json --random-input --mode=all" << endl;
    cout << "  facaded_cnn introspect --model=model.json --layer=0 --filter=0 --mode=feature_map" << endl;
    cout << "  facaded_cnn info --model=model.json" << endl;
    cout << "  facaded_cnn modify --model=model.json --action=add_filter --layer=0 --save=model_modified.json" << endl;
    cout << "  facaded_cnn export-onnx --model=model.json --onnx=model.onnx" << endl;
    cout << "  facaded_cnn import-onnx --onnx=model.onnx --save=imported.json" << endl;
    cout << endl;
}

enum TCommand {
    cmdNone, cmdCreate, cmdTrain, cmdPredict, cmdIntrospect, cmdStats,
    cmdModify, cmdAnalyze, cmdInfo, cmdHelp, cmdExportONNX, cmdImportONNX
};

int main(int argc, char* argv[]) {
    srand(time(NULL));

    if (argc < 2) {
        PrintUsage();
        return 0;
    }

    string CmdStr = argv[1];
    TCommand Command = cmdNone;

    if (CmdStr == "create") Command = cmdCreate;
    else if (CmdStr == "train") Command = cmdTrain;
    else if (CmdStr == "predict") Command = cmdPredict;
    else if (CmdStr == "introspect") Command = cmdIntrospect;
    else if (CmdStr == "stats") Command = cmdStats;
    else if (CmdStr == "modify") Command = cmdModify;
    else if (CmdStr == "analyze") Command = cmdAnalyze;
    else if (CmdStr == "info") Command = cmdInfo;
    else if (CmdStr == "export-onnx") Command = cmdExportONNX;
    else if (CmdStr == "import-onnx") Command = cmdImportONNX;
    else if (CmdStr == "help" || CmdStr == "--help" || CmdStr == "-h") Command = cmdHelp;
    else {
        cout << "Unknown command: " << CmdStr << endl;
        PrintUsage();
        return 1;
    }

    if (Command == cmdHelp) {
        PrintUsage();
        return 0;
    }

    int inputW = 0, inputH = 0, inputC = 0, outputSize = 0;
    vector<int> convFilters, kernelSizes, poolSizes, fcLayerSizes;
    double learningRate = 0.001, dropoutRate = 0.25;
    string modelFile = "", saveFile = "", predMode = "basic", onnxFile = "";
    int epochs = 20, layerIdx = 0, filterIdx = 0, channelIdx = 0;
    int histogramBins = 50, posX = 0, posY = 0;
    string intrinMode = "feature_map", statsType = "activations";
    string modifyAction = "", analyzeType = "receptive_field";
    bool randomInput = false;
    int filterCount = 1, kernelSize = 3, strideAdd = 1, paddingAdd = 0;

    for (int i = 2; i < argc; i++) {
        string arg = argv[i];
        size_t eqPos = arg.find('=');
        
        if (arg == "--random-input") {
            randomInput = true;
            continue;
        }
        
        if (eqPos == string::npos) continue;
        
        string key = arg.substr(0, eqPos);
        string value = arg.substr(eqPos + 1);

        if (key == "--input-w") inputW = stoi(value);
        else if (key == "--input-h") inputH = stoi(value);
        else if (key == "--input-c") inputC = stoi(value);
        else if (key == "--conv") ParseIntArrayHelper(value, convFilters);
        else if (key == "--kernels") ParseIntArrayHelper(value, kernelSizes);
        else if (key == "--pools") ParseIntArrayHelper(value, poolSizes);
        else if (key == "--fc") ParseIntArrayHelper(value, fcLayerSizes);
        else if (key == "--output") outputSize = stoi(value);
        else if (key == "--save") saveFile = value;
        else if (key == "--model") modelFile = value;
        else if (key == "--lr") learningRate = stod(value);
        else if (key == "--dropout") dropoutRate = stod(value);
        else if (key == "--epochs") epochs = stoi(value);
        else if (key == "--layer") layerIdx = stoi(value);
        else if (key == "--filter") filterIdx = stoi(value);
        else if (key == "--channel") channelIdx = stoi(value);
        else if (key == "--mode") {
            if (Command == cmdPredict) predMode = value;
            else if (Command == cmdIntrospect) intrinMode = value;
            else if (Command == cmdAnalyze) analyzeType = value;
        }
        else if (key == "--type") {
            if (Command == cmdStats) statsType = value;
            else if (Command == cmdAnalyze) analyzeType = value;
        }
        else if (key == "--histogram") histogramBins = stoi(value);
        else if (key == "--x") posX = stoi(value);
        else if (key == "--y") posY = stoi(value);
        else if (key == "--action") modifyAction = value;
        else if (key == "--filters") filterCount = stoi(value);
        else if (key == "--kernel") kernelSize = stoi(value);
        else if (key == "--stride") strideAdd = stoi(value);
        else if (key == "--padding") paddingAdd = stoi(value);
        else if (key == "--onnx") onnxFile = value;
    }

    TCNNFacade* CNN = nullptr;

    if (Command == cmdCreate) {
        if (inputW == 0 || inputH == 0 || inputC == 0 || outputSize == 0 ||
            convFilters.empty() || kernelSizes.empty() || poolSizes.empty() ||
            fcLayerSizes.empty() || saveFile.empty()) {
            cout << "Error: Missing required parameters for create command" << endl;
            PrintUsage();
            return 1;
        }

        cout << "Creating CNN model (OpenCL version)..." << endl;
        cout << "  Input: " << inputW << "x" << inputH << "x" << inputC << endl;
        cout << "  Conv filters: ";
        for (size_t i = 0; i < convFilters.size(); i++) {
            if (i > 0) cout << ", ";
            cout << convFilters[i];
        }
        cout << endl;
        cout << "  Output size: " << outputSize << endl;

        CNN = new TCNNFacade(inputW, inputH, inputC, convFilters, kernelSizes,
                             poolSizes, fcLayerSizes, outputSize, learningRate, dropoutRate);

        CNN->SaveModelToJSON(saveFile);
        cout << "Created CNN model" << endl;
        delete CNN;
    }
    else if (Command == cmdPredict) {
        if (modelFile.empty()) {
            cout << "Error: --model parameter required for predict command" << endl;
            PrintUsage();
            return 1;
        }

        cout << "Loading model from: " << modelFile << endl;
        CNN = new TCNNFacade(1, 1, 1, {}, {}, {}, {}, 1);
        CNN->LoadModelFromJSON(modelFile);

        int W = CNN->GetInputWidth();
        int H = CNN->GetInputHeight();
        int C = CNN->GetInputChannels();

        TImageData testImage;
        testImage.Width = W;
        testImage.Height = H;
        testImage.Channels = C;
        testImage.Data.resize(C, D2array(H, Darray(W)));

        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    testImage.Data[c][h][w] = (rand() % 100) / 100.0;
                }
            }
        }

        cout << "Making prediction..." << endl;
        Darray prediction = CNN->Predict(testImage);

        cout << "Prediction output (" << prediction.size() << " classes):" << endl;
        for (size_t i = 0; i < prediction.size(); i++) {
            cout << "  Class " << i << ": " << prediction[i] << endl;
        }

        delete CNN;
    }
    else if (Command == cmdInfo) {
        if (modelFile.empty()) {
            cout << "Error: --model parameter required for info command" << endl;
            PrintUsage();
            return 1;
        }

        cout << "Loading model from: " << modelFile << endl;
        CNN = new TCNNFacade(1, 1, 1, {}, {}, {}, {}, 1);
        CNN->LoadModelFromJSON(modelFile);

        cout << endl << "Model Architecture:" << endl;
        cout << "  Input: " << CNN->GetInputWidth() << "x"
             << CNN->GetInputHeight() << "x"
             << CNN->GetInputChannels() << endl;
        cout << "  Output size: " << CNN->GetOutputSize() << endl;
        cout << "  Learning rate: " << CNN->GetLearningRate() << endl;
        cout << "  Gradient clip: " << CNN->GetGradientClip() << endl;
        cout << "  activation: " << CNN->GetActivationTypeString() << endl;
        cout << "  output_activation: " << CNN->GetOutputActivationTypeString() << endl;
        cout << "  loss_type: " << CNN->GetLossType() << endl;
        cout << "  Total layers: " << CNN->GetNumLayers() << endl;

        delete CNN;
    }
    else if (Command == cmdExportONNX) {
        if (modelFile.empty()) {
            cout << "Error: --model parameter required for export-onnx command" << endl;
            PrintUsage();
            return 1;
        }
        if (onnxFile.empty()) {
            cout << "Error: --onnx parameter required for export-onnx command" << endl;
            PrintUsage();
            return 1;
        }

        cout << "Loading model from: " << modelFile << endl;
        CNN = new TCNNFacade(1, 1, 1, {}, {}, {}, {}, 1);
        CNN->LoadModelFromJSON(modelFile);
        CNN->ExportToONNX(onnxFile);
        delete CNN;
    }
    else if (Command == cmdImportONNX) {
        if (onnxFile.empty()) {
            cout << "Error: --onnx parameter required for import-onnx command" << endl;
            PrintUsage();
            return 1;
        }
        if (saveFile.empty()) {
            cout << "Error: --save parameter required for import-onnx command" << endl;
            PrintUsage();
            return 1;
        }

        CNN = TCNNFacade::ImportFromONNX(onnxFile);
        CNN->SaveModelToJSON(saveFile);
        cout << "Model imported from ONNX and saved to: " << saveFile << endl;
        delete CNN;
    }

    return 0;
}
