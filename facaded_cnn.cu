/*
  CNNFacade - CUDA Accelerated CNN Facade
  Matthew Abbott 2025
  Ported from Pascal to C++ to CUDA
  
  Compile: nvcc -O2 -o cnn_facade_cuda CNNFacadeCuda.cu
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define BLOCK_SIZE 16
#define TILE_SIZE 16

namespace CNNFacade {

constexpr double EPSILON = 1e-8;
constexpr double GRAD_CLIP = 1.0;

using Darray = std::vector<double>;
using D2array = std::vector<std::vector<double>>;
using D3array = std::vector<std::vector<std::vector<double>>>;
using D4array = std::vector<std::vector<std::vector<std::vector<double>>>>;
using IntArray = std::vector<int>;

// ============================================================================
// CUDA Kernels
// ============================================================================

__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void convForwardKernel(
    const double* input, const double* weights, const double* biases,
    double* preActivation, double* output,
    int inputChannels, int inputHeight, int inputWidth,
    int numFilters, int kernelSize, int stride, int padding,
    int outputHeight, int outputWidth)
{
    int f = blockIdx.z;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;

    if (f < numFilters && oh < outputHeight && ow < outputWidth) {
        double sum = biases[f];
        
        for (int c = 0; c < inputChannels; c++) {
            for (int kh = 0; kh < kernelSize; kh++) {
                for (int kw = 0; kw < kernelSize; kw++) {
                    int ih = oh * stride + kh - padding;
                    int iw = ow * stride + kw - padding;
                    
                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                        int inputIdx = c * inputHeight * inputWidth + ih * inputWidth + iw;
                        int weightIdx = f * inputChannels * kernelSize * kernelSize +
                                       c * kernelSize * kernelSize + kh * kernelSize + kw;
                        sum += input[inputIdx] * weights[weightIdx];
                    }
                }
            }
        }
        
        int outIdx = f * outputHeight * outputWidth + oh * outputWidth + ow;
        preActivation[outIdx] = sum;
        output[outIdx] = (sum > 0) ? sum : 0;  // ReLU
    }
}

__global__ void maxPoolForwardKernel(
    const double* input, double* output, int* maxIndices,
    int channels, int inputHeight, int inputWidth,
    int poolSize, int outputHeight, int outputWidth)
{
    int c = blockIdx.z;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;

    if (c < channels && oh < outputHeight && ow < outputWidth) {
        double maxVal = -1e308;
        int maxIdx = 0;
        
        for (int ph = 0; ph < poolSize; ph++) {
            for (int pw = 0; pw < poolSize; pw++) {
                int ih = oh * poolSize + ph;
                int iw = ow * poolSize + pw;
                int inputIdx = c * inputHeight * inputWidth + ih * inputWidth + iw;
                double val = input[inputIdx];
                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = ph * poolSize + pw;
                }
            }
        }
        
        int outIdx = c * outputHeight * outputWidth + oh * outputWidth + ow;
        output[outIdx] = maxVal;
        maxIndices[outIdx] = maxIdx;
    }
}

__global__ void fcForwardKernel(
    const double* input, const double* weights, const double* biases,
    double* preActivation, double* output, const double* dropoutMask,
    int numNeurons, int numInputs, bool applyRelu)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n < numNeurons) {
        double sum = biases[n];
        for (int i = 0; i < numInputs; i++) {
            sum += input[i] * weights[n * numInputs + i];
        }
        preActivation[n] = sum;
        if (applyRelu) {
            double relu = (sum > 0) ? sum : 0;
            output[n] = relu * dropoutMask[n];
        } else {
            output[n] = sum;
        }
    }
}

__global__ void softmaxKernel(double* logits, double* output, int size) {
    __shared__ double maxVal;
    __shared__ double sumExp;
    
    if (threadIdx.x == 0) {
        maxVal = -1e308;
        for (int i = 0; i < size; i++) {
            if (logits[i] > maxVal) maxVal = logits[i];
        }
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        sumExp = 0;
        for (int i = 0; i < size; i++) {
            double exp_val = exp(logits[i] - maxVal);
            output[i] = exp_val;
            sumExp += exp_val;
        }
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = output[idx] / sumExp;
        if (output[idx] < 1e-15) output[idx] = 1e-15;
        if (output[idx] > 1 - 1e-15) output[idx] = 1 - 1e-15;
    }
}

__global__ void convBackwardKernel(
    const double* gradOutput, const double* preActivation,
    const double* paddedInput, const double* weights,
    double* gradInput, double* gradWeights, double* gradBiases,
    int inputChannels, int inputHeight, int inputWidth,
    int numFilters, int kernelSize, int stride, int padding,
    int outputHeight, int outputWidth, int paddedHeight, int paddedWidth)
{
    int f = blockIdx.z;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;

    if (f < numFilters && oh < outputHeight && ow < outputWidth) {
        int outIdx = f * outputHeight * outputWidth + oh * outputWidth + ow;
        double grad = gradOutput[outIdx];
        
        // ReLU derivative
        if (preActivation[outIdx] <= 0) grad = 0;
        
        // Bias gradient
        atomicAddDouble(&gradBiases[f], grad);
        
        // Weight gradients
        for (int c = 0; c < inputChannels; c++) {
            for (int kh = 0; kh < kernelSize; kh++) {
                for (int kw = 0; kw < kernelSize; kw++) {
                    int ih = oh * stride + kh;
                    int iw = ow * stride + kw;
                    int inputIdx = c * paddedHeight * paddedWidth + ih * paddedWidth + iw;
                    int weightIdx = f * inputChannels * kernelSize * kernelSize +
                                   c * kernelSize * kernelSize + kh * kernelSize + kw;
                    atomicAddDouble(&gradWeights[weightIdx], grad * paddedInput[inputIdx]);
                }
            }
        }
        
        // Input gradients
        for (int c = 0; c < inputChannels; c++) {
            for (int kh = 0; kh < kernelSize; kh++) {
                for (int kw = 0; kw < kernelSize; kw++) {
                    int ih = oh * stride + kh - padding;
                    int iw = ow * stride + kw - padding;
                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                        int inputIdx = c * inputHeight * inputWidth + ih * inputWidth + iw;
                        int weightIdx = f * inputChannels * kernelSize * kernelSize +
                                       c * kernelSize * kernelSize + kh * kernelSize + kw;
                        atomicAddDouble(&gradInput[inputIdx], grad * weights[weightIdx]);
                    }
                }
            }
        }
    }
}

__global__ void maxPoolBackwardKernel(
    const double* gradOutput, const int* maxIndices, double* gradInput,
    int channels, int inputHeight, int inputWidth,
    int poolSize, int outputHeight, int outputWidth)
{
    int c = blockIdx.z;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;

    if (c < channels && oh < outputHeight && ow < outputWidth) {
        int outIdx = c * outputHeight * outputWidth + oh * outputWidth + ow;
        int maxIdx = maxIndices[outIdx];
        int maxH = maxIdx / poolSize;
        int maxW = maxIdx % poolSize;
        
        int ih = oh * poolSize + maxH;
        int iw = ow * poolSize + maxW;
        int inputIdx = c * inputHeight * inputWidth + ih * inputWidth + iw;
        
        atomicAddDouble(&gradInput[inputIdx], gradOutput[outIdx]);
    }
}

__global__ void fcBackwardKernel(
    const double* gradOutput, const double* preActivation,
    const double* input, const double* weights, const double* dropoutMask,
    double* gradInput, double* gradWeights, double* gradBiases,
    int numNeurons, int numInputs, bool isOutputLayer)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n < numNeurons) {
        double delta;
        if (isOutputLayer) {
            delta = gradOutput[n];
        } else {
            double reluDeriv = (preActivation[n] > 0) ? 1.0 : 0.0;
            delta = gradOutput[n] * reluDeriv * dropoutMask[n];
        }
        
        gradBiases[n] = delta;
        
        for (int i = 0; i < numInputs; i++) {
            atomicAddDouble(&gradInput[i], delta * weights[n * numInputs + i]);
            gradWeights[n * numInputs + i] = delta * input[i];
        }
    }
}

__global__ void adamUpdateKernel(
    double* weights, double* m, double* v, const double* grads,
    double lr, double beta1, double beta2, double epsilon,
    double beta1_t, double beta2_t, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double g = grads[idx];
        if (g > GRAD_CLIP) g = GRAD_CLIP;
        if (g < -GRAD_CLIP) g = -GRAD_CLIP;
        
        m[idx] = beta1 * m[idx] + (1 - beta1) * g;
        v[idx] = beta2 * v[idx] + (1 - beta2) * g * g;
        
        double m_hat = m[idx] / (1 - beta1_t);
        double v_hat = v[idx] / (1 - beta2_t);
        
        double update = lr * m_hat / (sqrt(v_hat) + epsilon);
        if (isfinite(update)) {
            weights[idx] -= update;
        }
    }
}

// ============================================================================
// Host Structures
// ============================================================================

struct TPoolIndex {
    int X = 0, Y = 0;
};
using TPoolIndexArray = std::vector<std::vector<std::vector<TPoolIndex>>>;

struct TLayerStats {
    double Mean = 0, StdDev = 0, Min = 0, Max = 0;
    int Count = 0;
};

struct TLayerConfig {
    std::string LayerType;
    int FilterCount = 0, KernelSize = 0, Stride = 0, Padding = 0;
    int InputChannels = 0, OutputWidth = 0, OutputHeight = 0;
    int PoolSize = 0, NeuronCount = 0, InputSize = 0;
};

struct TImageData {
    int Width = 0, Height = 0, Channels = 0;
    D3array Data;
};

struct TConvLayerGPU {
    double *d_weights, *d_biases;
    double *d_weightsM, *d_weightsV;
    double *d_biasM, *d_biasV;
    double *d_weightGrads, *d_biasGrads;
    double *d_output, *d_preActivation;
    double *d_input, *d_paddedInput;
    int numFilters, inputChannels, kernelSize, stride, padding;
    int inputHeight, inputWidth, outputHeight, outputWidth;
    int weightsSize, biasesSize, outputSize, inputSize, paddedSize;
};

struct TPoolLayerGPU {
    double *d_output;
    int *d_maxIndices;
    double *d_input;
    int channels, poolSize, stride;
    int inputHeight, inputWidth, outputHeight, outputWidth;
    int outputSize, inputSize;
};

struct TFCLayerGPU {
    double *d_weights, *d_biases;
    double *d_weightsM, *d_weightsV;
    double *d_biasM, *d_biasV;
    double *d_output, *d_preActivation;
    double *d_input, *d_dropoutMask;
    int numNeurons, numInputs;
    int weightsSize;
};

class TCNNFacade {
private:
    double LearningRate, DropoutRate, Beta1, Beta2;
    int AdamT;
    bool IsTraining;

    std::vector<TConvLayerGPU> ConvLayers;
    std::vector<TPoolLayerGPU> PoolLayers;
    std::vector<TFCLayerGPU> FullyConnectedLayers;
    TFCLayerGPU OutputLayer;

    int FlattenedSize;
    double* d_flattenedFeatures;
    int LastConvHeight, LastConvWidth, LastConvChannels;

    std::mt19937 rng;

    // Host-side copies for introspection
    std::vector<Darray> h_convWeights, h_convBiases;
    std::vector<Darray> h_fcWeights, h_fcBiases;
    Darray h_outputWeights, h_outputBiases;

    void InitializeConvLayer(TConvLayerGPU& layer, int numFilters, int inputChannels,
                             int kernelSize, int stride, int padding,
                             int inputHeight, int inputWidth) {
        layer.numFilters = numFilters;
        layer.inputChannels = inputChannels;
        layer.kernelSize = kernelSize;
        layer.stride = stride;
        layer.padding = padding;
        layer.inputHeight = inputHeight;
        layer.inputWidth = inputWidth;
        layer.outputHeight = (inputHeight + 2 * padding - kernelSize) / stride + 1;
        layer.outputWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;

        layer.weightsSize = numFilters * inputChannels * kernelSize * kernelSize;
        layer.biasesSize = numFilters;
        layer.outputSize = numFilters * layer.outputHeight * layer.outputWidth;
        layer.inputSize = inputChannels * inputHeight * inputWidth;
        int paddedH = inputHeight + 2 * padding;
        int paddedW = inputWidth + 2 * padding;
        layer.paddedSize = inputChannels * paddedH * paddedW;

        CUDA_CHECK(cudaMalloc(&layer.d_weights, layer.weightsSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_biases, layer.biasesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_weightsM, layer.weightsSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_weightsV, layer.weightsSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_biasM, layer.biasesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_biasV, layer.biasesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_weightGrads, layer.weightsSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_biasGrads, layer.biasesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_output, layer.outputSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_preActivation, layer.outputSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_input, layer.inputSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_paddedInput, layer.paddedSize * sizeof(double)));

        CUDA_CHECK(cudaMemset(layer.d_weightsM, 0, layer.weightsSize * sizeof(double)));
        CUDA_CHECK(cudaMemset(layer.d_weightsV, 0, layer.weightsSize * sizeof(double)));
        CUDA_CHECK(cudaMemset(layer.d_biasM, 0, layer.biasesSize * sizeof(double)));
        CUDA_CHECK(cudaMemset(layer.d_biasV, 0, layer.biasesSize * sizeof(double)));
        CUDA_CHECK(cudaMemset(layer.d_biases, 0, layer.biasesSize * sizeof(double)));

        // Initialize weights (He initialization)
        double scale = std::sqrt(2.0 / (inputChannels * kernelSize * kernelSize));
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        std::vector<double> h_weights(layer.weightsSize);
        for (int i = 0; i < layer.weightsSize; i++) h_weights[i] = dist(rng) * scale;
        CUDA_CHECK(cudaMemcpy(layer.d_weights, h_weights.data(), layer.weightsSize * sizeof(double), cudaMemcpyHostToDevice));
    }

    void InitializePoolLayer(TPoolLayerGPU& layer, int channels, int poolSize, int stride,
                             int inputHeight, int inputWidth) {
        layer.channels = channels;
        layer.poolSize = poolSize;
        layer.stride = stride;
        layer.inputHeight = inputHeight;
        layer.inputWidth = inputWidth;
        layer.outputHeight = inputHeight / poolSize;
        layer.outputWidth = inputWidth / poolSize;
        layer.outputSize = channels * layer.outputHeight * layer.outputWidth;
        layer.inputSize = channels * inputHeight * inputWidth;

        CUDA_CHECK(cudaMalloc(&layer.d_output, layer.outputSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_maxIndices, layer.outputSize * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&layer.d_input, layer.inputSize * sizeof(double)));
    }

    void InitializeFCLayer(TFCLayerGPU& layer, int numNeurons, int numInputs) {
        layer.numNeurons = numNeurons;
        layer.numInputs = numInputs;
        layer.weightsSize = numNeurons * numInputs;

        CUDA_CHECK(cudaMalloc(&layer.d_weights, layer.weightsSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_biases, numNeurons * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_weightsM, layer.weightsSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_weightsV, layer.weightsSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_biasM, numNeurons * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_biasV, numNeurons * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_output, numNeurons * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_preActivation, numNeurons * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_input, numInputs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&layer.d_dropoutMask, numNeurons * sizeof(double)));

        CUDA_CHECK(cudaMemset(layer.d_weightsM, 0, layer.weightsSize * sizeof(double)));
        CUDA_CHECK(cudaMemset(layer.d_weightsV, 0, layer.weightsSize * sizeof(double)));
        CUDA_CHECK(cudaMemset(layer.d_biasM, 0, numNeurons * sizeof(double)));
        CUDA_CHECK(cudaMemset(layer.d_biasV, 0, numNeurons * sizeof(double)));
        CUDA_CHECK(cudaMemset(layer.d_biases, 0, numNeurons * sizeof(double)));

        // Initialize weights
        double scale = std::sqrt(2.0 / numInputs);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        std::vector<double> h_weights(layer.weightsSize);
        for (int i = 0; i < layer.weightsSize; i++) h_weights[i] = dist(rng) * scale;
        CUDA_CHECK(cudaMemcpy(layer.d_weights, h_weights.data(), layer.weightsSize * sizeof(double), cudaMemcpyHostToDevice));

        // Initialize dropout mask to 1
        std::vector<double> h_mask(numNeurons, 1.0);
        CUDA_CHECK(cudaMemcpy(layer.d_dropoutMask, h_mask.data(), numNeurons * sizeof(double), cudaMemcpyHostToDevice));
    }

    void PadInput(const double* d_input, double* d_padded, int channels, int height, int width, int padding) {
        if (padding == 0) {
            CUDA_CHECK(cudaMemcpy(d_padded, d_input, channels * height * width * sizeof(double), cudaMemcpyDeviceToDevice));
            return;
        }
        int paddedH = height + 2 * padding;
        int paddedW = width + 2 * padding;
        CUDA_CHECK(cudaMemset(d_padded, 0, channels * paddedH * paddedW * sizeof(double)));
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                CUDA_CHECK(cudaMemcpy(
                    d_padded + c * paddedH * paddedW + (h + padding) * paddedW + padding,
                    d_input + c * height * width + h * width,
                    width * sizeof(double), cudaMemcpyDeviceToDevice));
            }
        }
    }

    void ApplyDropout(TFCLayerGPU& layer) {
        std::vector<double> h_mask(layer.numNeurons);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < layer.numNeurons; i++) {
            if (IsTraining && DropoutRate > 0) {
                h_mask[i] = (dist(rng) > DropoutRate) ? (1.0 / (1.0 - DropoutRate)) : 0.0;
            } else {
                h_mask[i] = 1.0;
            }
        }
        CUDA_CHECK(cudaMemcpy(layer.d_dropoutMask, h_mask.data(), layer.numNeurons * sizeof(double), cudaMemcpyHostToDevice));
    }

public:
    TCNNFacade(int InputWidth, int InputHeight, int InputChannels,
               const std::vector<int>& ConvFilters, const std::vector<int>& KernelSizes,
               const std::vector<int>& PoolSizes, const std::vector<int>& FCLayerSizes,
               int OutputSize, double ALearningRate = 0.001, double ADropoutRate = 0.25)
        : LearningRate(ALearningRate), DropoutRate(ADropoutRate), Beta1(0.9), Beta2(0.999),
          AdamT(0), IsTraining(false), FlattenedSize(0), d_flattenedFeatures(nullptr),
          LastConvHeight(0), LastConvWidth(0), LastConvChannels(0)
    {
        std::random_device rd;
        rng.seed(rd());

        int CurrentWidth = InputWidth, CurrentHeight = InputHeight, CurrentChannels = InputChannels;

        ConvLayers.resize(ConvFilters.size());
        PoolLayers.resize(PoolSizes.size());

        for (size_t i = 0; i < ConvFilters.size(); i++) {
            int kernelPadding = KernelSizes[i] / 2;
            InitializeConvLayer(ConvLayers[i], ConvFilters[i], CurrentChannels,
                               KernelSizes[i], 1, kernelPadding, CurrentHeight, CurrentWidth);
            CurrentWidth = ConvLayers[i].outputWidth;
            CurrentHeight = ConvLayers[i].outputHeight;
            CurrentChannels = ConvFilters[i];

            if (i < PoolSizes.size()) {
                InitializePoolLayer(PoolLayers[i], CurrentChannels, PoolSizes[i], PoolSizes[i],
                                   CurrentHeight, CurrentWidth);
                CurrentWidth = PoolLayers[i].outputWidth;
                CurrentHeight = PoolLayers[i].outputHeight;
            }
        }

        LastConvWidth = CurrentWidth;
        LastConvHeight = CurrentHeight;
        LastConvChannels = CurrentChannels;
        FlattenedSize = CurrentWidth * CurrentHeight * CurrentChannels;

        CUDA_CHECK(cudaMalloc(&d_flattenedFeatures, FlattenedSize * sizeof(double)));

        FullyConnectedLayers.resize(FCLayerSizes.size());
        int NumInputs = FlattenedSize;

        for (size_t i = 0; i < FCLayerSizes.size(); i++) {
            InitializeFCLayer(FullyConnectedLayers[i], FCLayerSizes[i], NumInputs);
            NumInputs = FCLayerSizes[i];
        }

        InitializeFCLayer(OutputLayer, OutputSize, NumInputs);
    }

    ~TCNNFacade() {
        for (auto& layer : ConvLayers) {
            cudaFree(layer.d_weights); cudaFree(layer.d_biases);
            cudaFree(layer.d_weightsM); cudaFree(layer.d_weightsV);
            cudaFree(layer.d_biasM); cudaFree(layer.d_biasV);
            cudaFree(layer.d_weightGrads); cudaFree(layer.d_biasGrads);
            cudaFree(layer.d_output); cudaFree(layer.d_preActivation);
            cudaFree(layer.d_input); cudaFree(layer.d_paddedInput);
        }
        for (auto& layer : PoolLayers) {
            cudaFree(layer.d_output); cudaFree(layer.d_maxIndices); cudaFree(layer.d_input);
        }
        for (auto& layer : FullyConnectedLayers) {
            cudaFree(layer.d_weights); cudaFree(layer.d_biases);
            cudaFree(layer.d_weightsM); cudaFree(layer.d_weightsV);
            cudaFree(layer.d_biasM); cudaFree(layer.d_biasV);
            cudaFree(layer.d_output); cudaFree(layer.d_preActivation);
            cudaFree(layer.d_input); cudaFree(layer.d_dropoutMask);
        }
        cudaFree(OutputLayer.d_weights); cudaFree(OutputLayer.d_biases);
        cudaFree(OutputLayer.d_weightsM); cudaFree(OutputLayer.d_weightsV);
        cudaFree(OutputLayer.d_biasM); cudaFree(OutputLayer.d_biasV);
        cudaFree(OutputLayer.d_output); cudaFree(OutputLayer.d_preActivation);
        cudaFree(OutputLayer.d_input); cudaFree(OutputLayer.d_dropoutMask);
        cudaFree(d_flattenedFeatures);
    }

    Darray Predict(TImageData& Image) {
        if (Image.Data.empty()) return Darray(OutputLayer.numNeurons, 0);

        // Flatten input image to device
        std::vector<double> h_input(Image.Channels * Image.Height * Image.Width);
        int idx = 0;
        for (int c = 0; c < Image.Channels; c++)
            for (int h = 0; h < Image.Height; h++)
                for (int w = 0; w < Image.Width; w++)
                    h_input[idx++] = Image.Data[c][h][w];

        double* d_currentInput;
        int currentSize = h_input.size();
        CUDA_CHECK(cudaMalloc(&d_currentInput, currentSize * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_currentInput, h_input.data(), currentSize * sizeof(double), cudaMemcpyHostToDevice));

        int CurrentWidth = Image.Width, CurrentHeight = Image.Height;

        // Conv + Pool forward
        for (size_t i = 0; i < ConvLayers.size(); i++) {
            auto& conv = ConvLayers[i];
            
            CUDA_CHECK(cudaMemcpy(conv.d_input, d_currentInput, conv.inputSize * sizeof(double), cudaMemcpyDeviceToDevice));
            int paddedH = conv.inputHeight + 2 * conv.padding;
            int paddedW = conv.inputWidth + 2 * conv.padding;
            PadInput(conv.d_input, conv.d_paddedInput, conv.inputChannels, conv.inputHeight, conv.inputWidth, conv.padding);

            dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
            dim3 gridSize((conv.outputWidth + BLOCK_SIZE - 1) / BLOCK_SIZE,
                         (conv.outputHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
                         conv.numFilters);

            convForwardKernel<<<gridSize, blockSize>>>(
                conv.d_paddedInput, conv.d_weights, conv.d_biases,
                conv.d_preActivation, conv.d_output,
                conv.inputChannels, paddedH, paddedW,
                conv.numFilters, conv.kernelSize, conv.stride, 0,
                conv.outputHeight, conv.outputWidth);
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaFree(d_currentInput);
            CurrentWidth = conv.outputWidth;
            CurrentHeight = conv.outputHeight;

            if (i < PoolLayers.size()) {
                auto& pool = PoolLayers[i];
                CUDA_CHECK(cudaMemcpy(pool.d_input, conv.d_output, pool.inputSize * sizeof(double), cudaMemcpyDeviceToDevice));

                dim3 poolGrid((pool.outputWidth + BLOCK_SIZE - 1) / BLOCK_SIZE,
                             (pool.outputHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
                             pool.channels);

                maxPoolForwardKernel<<<poolGrid, blockSize>>>(
                    pool.d_input, pool.d_output, pool.d_maxIndices,
                    pool.channels, pool.inputHeight, pool.inputWidth,
                    pool.poolSize, pool.outputHeight, pool.outputWidth);
                CUDA_CHECK(cudaDeviceSynchronize());

                CUDA_CHECK(cudaMalloc(&d_currentInput, pool.outputSize * sizeof(double)));
                CUDA_CHECK(cudaMemcpy(d_currentInput, pool.d_output, pool.outputSize * sizeof(double), cudaMemcpyDeviceToDevice));
                CurrentWidth = pool.outputWidth;
                CurrentHeight = pool.outputHeight;
            } else {
                CUDA_CHECK(cudaMalloc(&d_currentInput, conv.outputSize * sizeof(double)));
                CUDA_CHECK(cudaMemcpy(d_currentInput, conv.d_output, conv.outputSize * sizeof(double), cudaMemcpyDeviceToDevice));
            }
        }

        // Flatten
        CUDA_CHECK(cudaMemcpy(d_flattenedFeatures, d_currentInput, FlattenedSize * sizeof(double), cudaMemcpyDeviceToDevice));
        cudaFree(d_currentInput);

        // FC forward
        double* d_layerInput = d_flattenedFeatures;
        for (size_t i = 0; i < FullyConnectedLayers.size(); i++) {
            auto& fc = FullyConnectedLayers[i];
            ApplyDropout(fc);
            CUDA_CHECK(cudaMemcpy(fc.d_input, d_layerInput, fc.numInputs * sizeof(double), cudaMemcpyDeviceToDevice));

            int blocks = (fc.numNeurons + 255) / 256;
            fcForwardKernel<<<blocks, 256>>>(
                fc.d_input, fc.d_weights, fc.d_biases,
                fc.d_preActivation, fc.d_output, fc.d_dropoutMask,
                fc.numNeurons, fc.numInputs, true);
            CUDA_CHECK(cudaDeviceSynchronize());

            d_layerInput = fc.d_output;
        }

        // Output layer
        CUDA_CHECK(cudaMemcpy(OutputLayer.d_input, d_layerInput, OutputLayer.numInputs * sizeof(double), cudaMemcpyDeviceToDevice));
        int blocks = (OutputLayer.numNeurons + 255) / 256;
        fcForwardKernel<<<blocks, 256>>>(
            OutputLayer.d_input, OutputLayer.d_weights, OutputLayer.d_biases,
            OutputLayer.d_preActivation, OutputLayer.d_output, OutputLayer.d_dropoutMask,
            OutputLayer.numNeurons, OutputLayer.numInputs, false);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Softmax
        softmaxKernel<<<1, 256>>>(OutputLayer.d_preActivation, OutputLayer.d_output, OutputLayer.numNeurons);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back
        Darray result(OutputLayer.numNeurons);
        CUDA_CHECK(cudaMemcpy(result.data(), OutputLayer.d_output, OutputLayer.numNeurons * sizeof(double), cudaMemcpyDeviceToHost));

        return result;
    }

    double TrainStep(TImageData& Image, const Darray& Target) {
        IsTraining = true;
        Darray prediction = Predict(Image);

        // Compute output gradient
        std::vector<double> h_outputGrad(OutputLayer.numNeurons);
        for (int i = 0; i < OutputLayer.numNeurons; i++)
            h_outputGrad[i] = prediction[i] - Target[i];

        double* d_outputGrad;
        CUDA_CHECK(cudaMalloc(&d_outputGrad, OutputLayer.numNeurons * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_outputGrad, h_outputGrad.data(), OutputLayer.numNeurons * sizeof(double), cudaMemcpyHostToDevice));

        // Allocate gradient buffers
        double* d_fcGrad;
        CUDA_CHECK(cudaMalloc(&d_fcGrad, OutputLayer.numInputs * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_fcGrad, 0, OutputLayer.numInputs * sizeof(double)));

        double* d_weightGrads;
        double* d_biasGrads;
        CUDA_CHECK(cudaMalloc(&d_weightGrads, OutputLayer.weightsSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_biasGrads, OutputLayer.numNeurons * sizeof(double)));

        // Output layer backward
        int blocks = (OutputLayer.numNeurons + 255) / 256;
        fcBackwardKernel<<<blocks, 256>>>(
            d_outputGrad, OutputLayer.d_preActivation,
            OutputLayer.d_input, OutputLayer.d_weights, OutputLayer.d_dropoutMask,
            d_fcGrad, d_weightGrads, d_biasGrads,
            OutputLayer.numNeurons, OutputLayer.numInputs, true);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update output layer weights
        AdamT++;
        double beta1_t = std::pow(Beta1, AdamT);
        double beta2_t = std::pow(Beta2, AdamT);

        int updateBlocks = (OutputLayer.weightsSize + 255) / 256;
        adamUpdateKernel<<<updateBlocks, 256>>>(
            OutputLayer.d_weights, OutputLayer.d_weightsM, OutputLayer.d_weightsV, d_weightGrads,
            LearningRate, Beta1, Beta2, EPSILON, beta1_t, beta2_t, OutputLayer.weightsSize);

        updateBlocks = (OutputLayer.numNeurons + 255) / 256;
        adamUpdateKernel<<<updateBlocks, 256>>>(
            OutputLayer.d_biases, OutputLayer.d_biasM, OutputLayer.d_biasV, d_biasGrads,
            LearningRate, Beta1, Beta2, EPSILON, beta1_t, beta2_t, OutputLayer.numNeurons);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(d_weightGrads);
        cudaFree(d_biasGrads);
        cudaFree(d_outputGrad);

        // FC layers backward
        for (int i = FullyConnectedLayers.size() - 1; i >= 0; i--) {
            auto& fc = FullyConnectedLayers[i];
            double* d_nextGrad;
            CUDA_CHECK(cudaMalloc(&d_nextGrad, fc.numInputs * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_nextGrad, 0, fc.numInputs * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_weightGrads, fc.weightsSize * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_biasGrads, fc.numNeurons * sizeof(double)));

            blocks = (fc.numNeurons + 255) / 256;
            fcBackwardKernel<<<blocks, 256>>>(
                d_fcGrad, fc.d_preActivation, fc.d_input, fc.d_weights, fc.d_dropoutMask,
                d_nextGrad, d_weightGrads, d_biasGrads,
                fc.numNeurons, fc.numInputs, false);
            CUDA_CHECK(cudaDeviceSynchronize());

            updateBlocks = (fc.weightsSize + 255) / 256;
            adamUpdateKernel<<<updateBlocks, 256>>>(
                fc.d_weights, fc.d_weightsM, fc.d_weightsV, d_weightGrads,
                LearningRate, Beta1, Beta2, EPSILON, beta1_t, beta2_t, fc.weightsSize);

            updateBlocks = (fc.numNeurons + 255) / 256;
            adamUpdateKernel<<<updateBlocks, 256>>>(
                fc.d_biases, fc.d_biasM, fc.d_biasV, d_biasGrads,
                LearningRate, Beta1, Beta2, EPSILON, beta1_t, beta2_t, fc.numNeurons);
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaFree(d_fcGrad);
            cudaFree(d_weightGrads);
            cudaFree(d_biasGrads);
            d_fcGrad = d_nextGrad;
        }

        // Unflatten gradient to conv
        double* d_convGrad = d_fcGrad;

        // Conv layers backward
        for (int i = ConvLayers.size() - 1; i >= 0; i--) {
            // Pool backward
            if (i < (int)PoolLayers.size()) {
                auto& pool = PoolLayers[i];
                double* d_poolGradIn;
                CUDA_CHECK(cudaMalloc(&d_poolGradIn, pool.inputSize * sizeof(double)));
                CUDA_CHECK(cudaMemset(d_poolGradIn, 0, pool.inputSize * sizeof(double)));

                dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
                dim3 gridSize((pool.outputWidth + BLOCK_SIZE - 1) / BLOCK_SIZE,
                             (pool.outputHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
                             pool.channels);

                maxPoolBackwardKernel<<<gridSize, blockSize>>>(
                    d_convGrad, pool.d_maxIndices, d_poolGradIn,
                    pool.channels, pool.inputHeight, pool.inputWidth,
                    pool.poolSize, pool.outputHeight, pool.outputWidth);
                CUDA_CHECK(cudaDeviceSynchronize());

                cudaFree(d_convGrad);
                d_convGrad = d_poolGradIn;
            }

            // Conv backward
            auto& conv = ConvLayers[i];
            double* d_convGradIn;
            CUDA_CHECK(cudaMalloc(&d_convGradIn, conv.inputSize * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_convGradIn, 0, conv.inputSize * sizeof(double)));
            CUDA_CHECK(cudaMemset(conv.d_weightGrads, 0, conv.weightsSize * sizeof(double)));
            CUDA_CHECK(cudaMemset(conv.d_biasGrads, 0, conv.biasesSize * sizeof(double)));

            int paddedH = conv.inputHeight + 2 * conv.padding;
            int paddedW = conv.inputWidth + 2 * conv.padding;

            dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
            dim3 gridSize((conv.outputWidth + BLOCK_SIZE - 1) / BLOCK_SIZE,
                         (conv.outputHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
                         conv.numFilters);

            convBackwardKernel<<<gridSize, blockSize>>>(
                d_convGrad, conv.d_preActivation, conv.d_paddedInput, conv.d_weights,
                d_convGradIn, conv.d_weightGrads, conv.d_biasGrads,
                conv.inputChannels, conv.inputHeight, conv.inputWidth,
                conv.numFilters, conv.kernelSize, conv.stride, conv.padding,
                conv.outputHeight, conv.outputWidth, paddedH, paddedW);
            CUDA_CHECK(cudaDeviceSynchronize());

            updateBlocks = (conv.weightsSize + 255) / 256;
            adamUpdateKernel<<<updateBlocks, 256>>>(
                conv.d_weights, conv.d_weightsM, conv.d_weightsV, conv.d_weightGrads,
                LearningRate, Beta1, Beta2, EPSILON, beta1_t, beta2_t, conv.weightsSize);

            updateBlocks = (conv.biasesSize + 255) / 256;
            adamUpdateKernel<<<updateBlocks, 256>>>(
                conv.d_biases, conv.d_biasM, conv.d_biasV, conv.d_biasGrads,
                LearningRate, Beta1, Beta2, EPSILON, beta1_t, beta2_t, conv.biasesSize);
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaFree(d_convGrad);
            d_convGrad = d_convGradIn;
        }

        cudaFree(d_convGrad);

        // Cross-entropy loss
        double loss = 0;
        for (int i = 0; i < (int)Target.size(); i++) {
            if (Target[i] > 0) {
                double p = std::max(1e-15, std::min(1 - 1e-15, prediction[i]));
                loss -= Target[i] * std::log(p);
            }
        }

        return loss;
    }

    void SaveModel(const std::string& Filename) {
        std::ofstream file(Filename, std::ios::binary);
        if (!file) return;

        auto writeInt = [&](int v) { file.write(reinterpret_cast<char*>(&v), sizeof(v)); };
        auto writeDouble = [&](double v) { file.write(reinterpret_cast<char*>(&v), sizeof(v)); };

        writeInt(ConvLayers.size());
        for (auto& layer : ConvLayers) {
            writeInt(layer.numFilters);
            writeInt(layer.kernelSize);
            writeInt(layer.stride);
            writeInt(layer.padding);
            writeInt(layer.inputChannels);
            std::vector<double> h_weights(layer.weightsSize);
            std::vector<double> h_biases(layer.biasesSize);
            CUDA_CHECK(cudaMemcpy(h_weights.data(), layer.d_weights, layer.weightsSize * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_biases.data(), layer.d_biases, layer.biasesSize * sizeof(double), cudaMemcpyDeviceToHost));
            for (double b : h_biases) writeDouble(b);
            for (double w : h_weights) writeDouble(w);
        }

        writeInt(FullyConnectedLayers.size());
        for (auto& layer : FullyConnectedLayers) {
            writeInt(layer.numNeurons);
            writeInt(layer.numInputs);
            std::vector<double> h_weights(layer.weightsSize);
            std::vector<double> h_biases(layer.numNeurons);
            CUDA_CHECK(cudaMemcpy(h_weights.data(), layer.d_weights, layer.weightsSize * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_biases.data(), layer.d_biases, layer.numNeurons * sizeof(double), cudaMemcpyDeviceToHost));
            for (double b : h_biases) writeDouble(b);
            for (double w : h_weights) writeDouble(w);
        }

        writeInt(OutputLayer.numNeurons);
        writeInt(OutputLayer.numInputs);
        std::vector<double> h_weights(OutputLayer.weightsSize);
        std::vector<double> h_biases(OutputLayer.numNeurons);
        CUDA_CHECK(cudaMemcpy(h_weights.data(), OutputLayer.d_weights, OutputLayer.weightsSize * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_biases.data(), OutputLayer.d_biases, OutputLayer.numNeurons * sizeof(double), cudaMemcpyDeviceToHost));
        for (double b : h_biases) writeDouble(b);
        for (double w : h_weights) writeDouble(w);
    }

    void LoadModel(const std::string& Filename) {
        std::ifstream file(Filename, std::ios::binary);
        if (!file) return;

        auto readInt = [&]() { int v; file.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; };
        auto readDouble = [&]() { double v; file.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; };

        int numConv = readInt();
        for (int l = 0; l < numConv && l < (int)ConvLayers.size(); l++) {
            auto& layer = ConvLayers[l];
            readInt(); readInt(); readInt(); readInt(); readInt(); // skip config
            std::vector<double> h_biases(layer.biasesSize);
            std::vector<double> h_weights(layer.weightsSize);
            for (int i = 0; i < layer.biasesSize; i++) h_biases[i] = readDouble();
            for (int i = 0; i < layer.weightsSize; i++) h_weights[i] = readDouble();
            CUDA_CHECK(cudaMemcpy(layer.d_biases, h_biases.data(), layer.biasesSize * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(layer.d_weights, h_weights.data(), layer.weightsSize * sizeof(double), cudaMemcpyHostToDevice));
        }

        int numFC = readInt();
        for (int l = 0; l < numFC && l < (int)FullyConnectedLayers.size(); l++) {
            auto& layer = FullyConnectedLayers[l];
            readInt(); readInt(); // skip config
            std::vector<double> h_biases(layer.numNeurons);
            std::vector<double> h_weights(layer.weightsSize);
            for (int i = 0; i < layer.numNeurons; i++) h_biases[i] = readDouble();
            for (int i = 0; i < layer.weightsSize; i++) h_weights[i] = readDouble();
            CUDA_CHECK(cudaMemcpy(layer.d_biases, h_biases.data(), layer.numNeurons * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(layer.d_weights, h_weights.data(), layer.weightsSize * sizeof(double), cudaMemcpyHostToDevice));
        }

        readInt(); readInt(); // skip output config
        std::vector<double> h_biases(OutputLayer.numNeurons);
        std::vector<double> h_weights(OutputLayer.weightsSize);
        for (int i = 0; i < OutputLayer.numNeurons; i++) h_biases[i] = readDouble();
        for (int i = 0; i < OutputLayer.weightsSize; i++) h_weights[i] = readDouble();
        CUDA_CHECK(cudaMemcpy(OutputLayer.d_biases, h_biases.data(), OutputLayer.numNeurons * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(OutputLayer.d_weights, h_weights.data(), OutputLayer.weightsSize * sizeof(double), cudaMemcpyHostToDevice));
    }

    int GetNumLayers() { return ConvLayers.size() + FullyConnectedLayers.size() + 1; }
    int GetNumConvLayers() { return ConvLayers.size(); }
    int GetNumFCLayers() { return FullyConnectedLayers.size() + 1; }

    TLayerConfig GetLayerConfig(int LayerIdx) {
        TLayerConfig cfg;
        if (LayerIdx >= 0 && LayerIdx < (int)ConvLayers.size()) {
            cfg.LayerType = "conv";
            cfg.FilterCount = ConvLayers[LayerIdx].numFilters;
            cfg.KernelSize = ConvLayers[LayerIdx].kernelSize;
            cfg.Stride = ConvLayers[LayerIdx].stride;
            cfg.Padding = ConvLayers[LayerIdx].padding;
            cfg.InputChannels = ConvLayers[LayerIdx].inputChannels;
            cfg.OutputHeight = ConvLayers[LayerIdx].outputHeight;
            cfg.OutputWidth = ConvLayers[LayerIdx].outputWidth;
        } else {
            int fcIdx = LayerIdx - ConvLayers.size();
            if (fcIdx >= 0 && fcIdx < (int)FullyConnectedLayers.size()) {
                cfg.LayerType = "fc";
                cfg.NeuronCount = FullyConnectedLayers[fcIdx].numNeurons;
                cfg.InputSize = FullyConnectedLayers[fcIdx].numInputs;
            } else if (fcIdx == (int)FullyConnectedLayers.size()) {
                cfg.LayerType = "output";
                cfg.NeuronCount = OutputLayer.numNeurons;
                cfg.InputSize = OutputLayer.numInputs;
            }
        }
        return cfg;
    }

    void SetTrainingMode(bool mode) { IsTraining = mode; }
    bool GetTrainingMode() { return IsTraining; }

    // Stage 1: Feature Map Access
    D2array GetFeatureMap(int LayerIdx, int FilterIdx) {
        D2array result;
        if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return result;
        auto& conv = ConvLayers[LayerIdx];
        if (FilterIdx < 0 || FilterIdx >= conv.numFilters) return result;
        int h = conv.outputHeight, w = conv.outputWidth;
        result.resize(h, Darray(w));
        std::vector<double> h_output(conv.outputSize);
        CUDA_CHECK(cudaMemcpy(h_output.data(), conv.d_output, conv.outputSize * sizeof(double), cudaMemcpyDeviceToHost));
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                result[y][x] = h_output[FilterIdx * h * w + y * w + x];
        return result;
    }

    D2array GetPreActivation(int LayerIdx, int FilterIdx) {
        D2array result;
        if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return result;
        auto& conv = ConvLayers[LayerIdx];
        if (FilterIdx < 0 || FilterIdx >= conv.numFilters) return result;
        int h = conv.outputHeight, w = conv.outputWidth;
        result.resize(h, Darray(w));
        std::vector<double> h_pre(conv.outputSize);
        CUDA_CHECK(cudaMemcpy(h_pre.data(), conv.d_preActivation, conv.outputSize * sizeof(double), cudaMemcpyDeviceToHost));
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                result[y][x] = h_pre[FilterIdx * h * w + y * w + x];
        return result;
    }

    // Stage 2: Kernel/Filter Access
    D3array GetKernel(int LayerIdx, int FilterIdx) {
        D3array result;
        if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return result;
        auto& conv = ConvLayers[LayerIdx];
        if (FilterIdx < 0 || FilterIdx >= conv.numFilters) return result;
        int ic = conv.inputChannels, ks = conv.kernelSize;
        result.resize(ic, D2array(ks, Darray(ks)));
        std::vector<double> h_weights(conv.weightsSize);
        CUDA_CHECK(cudaMemcpy(h_weights.data(), conv.d_weights, conv.weightsSize * sizeof(double), cudaMemcpyDeviceToHost));
        int offset = FilterIdx * ic * ks * ks;
        for (int c = 0; c < ic; c++)
            for (int kh = 0; kh < ks; kh++)
                for (int kw = 0; kw < ks; kw++)
                    result[c][kh][kw] = h_weights[offset + c * ks * ks + kh * ks + kw];
        return result;
    }

    void SetKernel(int LayerIdx, int FilterIdx, const D3array& kernel) {
        if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return;
        auto& conv = ConvLayers[LayerIdx];
        if (FilterIdx < 0 || FilterIdx >= conv.numFilters) return;
        int ic = conv.inputChannels, ks = conv.kernelSize;
        std::vector<double> h_weights(conv.weightsSize);
        CUDA_CHECK(cudaMemcpy(h_weights.data(), conv.d_weights, conv.weightsSize * sizeof(double), cudaMemcpyDeviceToHost));
        int offset = FilterIdx * ic * ks * ks;
        for (int c = 0; c < ic && c < (int)kernel.size(); c++)
            for (int kh = 0; kh < ks && kh < (int)kernel[c].size(); kh++)
                for (int kw = 0; kw < ks && kw < (int)kernel[c][kh].size(); kw++)
                    h_weights[offset + c * ks * ks + kh * ks + kw] = kernel[c][kh][kw];
        CUDA_CHECK(cudaMemcpy(conv.d_weights, h_weights.data(), conv.weightsSize * sizeof(double), cudaMemcpyHostToDevice));
    }

    double GetBias(int LayerIdx, int FilterIdx) {
        if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return 0;
        auto& conv = ConvLayers[LayerIdx];
        if (FilterIdx < 0 || FilterIdx >= conv.numFilters) return 0;
        std::vector<double> h_biases(conv.biasesSize);
        CUDA_CHECK(cudaMemcpy(h_biases.data(), conv.d_biases, conv.biasesSize * sizeof(double), cudaMemcpyDeviceToHost));
        return h_biases[FilterIdx];
    }

    void SetBias(int LayerIdx, int FilterIdx, double value) {
        if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return;
        auto& conv = ConvLayers[LayerIdx];
        if (FilterIdx < 0 || FilterIdx >= conv.numFilters) return;
        std::vector<double> h_biases(conv.biasesSize);
        CUDA_CHECK(cudaMemcpy(h_biases.data(), conv.d_biases, conv.biasesSize * sizeof(double), cudaMemcpyDeviceToHost));
        h_biases[FilterIdx] = value;
        CUDA_CHECK(cudaMemcpy(conv.d_biases, h_biases.data(), conv.biasesSize * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Stage 4: Pooling Indices
    D2array GetPoolingIndices(int LayerIdx, int FilterIdx) {
        D2array result;
        if (LayerIdx < 0 || LayerIdx >= (int)PoolLayers.size()) return result;
        auto& pool = PoolLayers[LayerIdx];
        if (FilterIdx < 0 || FilterIdx >= pool.channels) return result;
        int h = pool.outputHeight, w = pool.outputWidth;
        result.resize(h, Darray(w));
        std::vector<int> h_indices(pool.outputSize);
        CUDA_CHECK(cudaMemcpy(h_indices.data(), pool.d_maxIndices, pool.outputSize * sizeof(int), cudaMemcpyDeviceToHost));
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                result[y][x] = h_indices[FilterIdx * h * w + y * w + x];
        return result;
    }

    // Stage 5: Gradients
    D3array GetFilterGradient(int LayerIdx, int FilterIdx) {
        D3array result;
        if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return result;
        auto& conv = ConvLayers[LayerIdx];
        if (FilterIdx < 0 || FilterIdx >= conv.numFilters) return result;
        int ic = conv.inputChannels, ks = conv.kernelSize;
        result.resize(ic, D2array(ks, Darray(ks)));
        std::vector<double> h_grads(conv.weightsSize);
        CUDA_CHECK(cudaMemcpy(h_grads.data(), conv.d_weightGrads, conv.weightsSize * sizeof(double), cudaMemcpyDeviceToHost));
        int offset = FilterIdx * ic * ks * ks;
        for (int c = 0; c < ic; c++)
            for (int kh = 0; kh < ks; kh++)
                for (int kw = 0; kw < ks; kw++)
                    result[c][kh][kw] = h_grads[offset + c * ks * ks + kh * ks + kw];
        return result;
    }

    double GetBiasGradient(int LayerIdx, int FilterIdx) {
        if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return 0;
        auto& conv = ConvLayers[LayerIdx];
        if (FilterIdx < 0 || FilterIdx >= conv.numFilters) return 0;
        std::vector<double> h_grads(conv.biasesSize);
        CUDA_CHECK(cudaMemcpy(h_grads.data(), conv.d_biasGrads, conv.biasesSize * sizeof(double), cudaMemcpyDeviceToHost));
        return h_grads[FilterIdx];
    }

    // Stage 6: Flattened Features
    Darray GetFlattenedFeatures() {
        Darray result(FlattenedSize);
        CUDA_CHECK(cudaMemcpy(result.data(), d_flattenedFeatures, FlattenedSize * sizeof(double), cudaMemcpyDeviceToHost));
        return result;
    }

    // Stage 7: Output Layer
    Darray GetLogits() {
        Darray result(OutputLayer.numNeurons);
        CUDA_CHECK(cudaMemcpy(result.data(), OutputLayer.d_preActivation, OutputLayer.numNeurons * sizeof(double), cudaMemcpyDeviceToHost));
        return result;
    }

    Darray GetSoftmaxOutput() {
        Darray result(OutputLayer.numNeurons);
        CUDA_CHECK(cudaMemcpy(result.data(), OutputLayer.d_output, OutputLayer.numNeurons * sizeof(double), cudaMemcpyDeviceToHost));
        return result;
    }

    // Stage 8: FC Layer Access
    Darray GetFCWeights(int LayerIdx, int NeuronIdx) {
        Darray result;
        TFCLayerGPU* layer = nullptr;
        if (LayerIdx >= 0 && LayerIdx < (int)FullyConnectedLayers.size())
            layer = &FullyConnectedLayers[LayerIdx];
        else if (LayerIdx == (int)FullyConnectedLayers.size())
            layer = &OutputLayer;
        if (!layer || NeuronIdx < 0 || NeuronIdx >= layer->numNeurons) return result;
        result.resize(layer->numInputs);
        std::vector<double> h_weights(layer->weightsSize);
        CUDA_CHECK(cudaMemcpy(h_weights.data(), layer->d_weights, layer->weightsSize * sizeof(double), cudaMemcpyDeviceToHost));
        for (int i = 0; i < layer->numInputs; i++)
            result[i] = h_weights[NeuronIdx * layer->numInputs + i];
        return result;
    }

    double GetFCBias(int LayerIdx, int NeuronIdx) {
        TFCLayerGPU* layer = nullptr;
        if (LayerIdx >= 0 && LayerIdx < (int)FullyConnectedLayers.size())
            layer = &FullyConnectedLayers[LayerIdx];
        else if (LayerIdx == (int)FullyConnectedLayers.size())
            layer = &OutputLayer;
        if (!layer || NeuronIdx < 0 || NeuronIdx >= layer->numNeurons) return 0;
        std::vector<double> h_biases(layer->numNeurons);
        CUDA_CHECK(cudaMemcpy(h_biases.data(), layer->d_biases, layer->numNeurons * sizeof(double), cudaMemcpyDeviceToHost));
        return h_biases[NeuronIdx];
    }

    void SetFCBias(int LayerIdx, int NeuronIdx, double value) {
        TFCLayerGPU* layer = nullptr;
        if (LayerIdx >= 0 && LayerIdx < (int)FullyConnectedLayers.size())
            layer = &FullyConnectedLayers[LayerIdx];
        else if (LayerIdx == (int)FullyConnectedLayers.size())
            layer = &OutputLayer;
        if (!layer || NeuronIdx < 0 || NeuronIdx >= layer->numNeurons) return;
        std::vector<double> h_biases(layer->numNeurons);
        CUDA_CHECK(cudaMemcpy(h_biases.data(), layer->d_biases, layer->numNeurons * sizeof(double), cudaMemcpyDeviceToHost));
        h_biases[NeuronIdx] = value;
        CUDA_CHECK(cudaMemcpy(layer->d_biases, h_biases.data(), layer->numNeurons * sizeof(double), cudaMemcpyHostToDevice));
    }

    Darray GetDropoutMask(int LayerIdx) {
        Darray result;
        if (LayerIdx < 0 || LayerIdx >= (int)FullyConnectedLayers.size()) return result;
        auto& fc = FullyConnectedLayers[LayerIdx];
        result.resize(fc.numNeurons);
        CUDA_CHECK(cudaMemcpy(result.data(), fc.d_dropoutMask, fc.numNeurons * sizeof(double), cudaMemcpyDeviceToHost));
        return result;
    }

    // Stage 11: Statistics
    TLayerStats GetLayerStats(int LayerIdx) {
        TLayerStats stats;
        if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return stats;
        auto& conv = ConvLayers[LayerIdx];
        std::vector<double> h_output(conv.outputSize);
        CUDA_CHECK(cudaMemcpy(h_output.data(), conv.d_output, conv.outputSize * sizeof(double), cudaMemcpyDeviceToHost));
        stats.Min = 1e308; stats.Max = -1e308;
        double sum = 0, sumSq = 0;
        stats.Count = conv.outputSize;
        for (int i = 0; i < conv.outputSize; i++) {
            double v = h_output[i];
            if (v < stats.Min) stats.Min = v;
            if (v > stats.Max) stats.Max = v;
            sum += v;
            sumSq += v * v;
        }
        if (stats.Count > 0) {
            stats.Mean = sum / stats.Count;
            stats.StdDev = std::sqrt((sumSq / stats.Count) - (stats.Mean * stats.Mean));
        }
        return stats;
    }

    Darray GetActivationHistogram(int LayerIdx, int numBins = 50) {
        Darray result(numBins, 0);
        if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return result;
        auto& conv = ConvLayers[LayerIdx];
        std::vector<double> h_output(conv.outputSize);
        CUDA_CHECK(cudaMemcpy(h_output.data(), conv.d_output, conv.outputSize * sizeof(double), cudaMemcpyDeviceToHost));
        double minVal = 1e308, maxVal = -1e308;
        for (double v : h_output) { if (v < minVal) minVal = v; if (v > maxVal) maxVal = v; }
        double range = maxVal - minVal;
        if (range < 1e-8) range = 1;
        for (double v : h_output) {
            int bin = (int)((v - minVal) / range * (numBins - 1));
            if (bin >= numBins) bin = numBins - 1;
            if (bin < 0) bin = 0;
            result[bin]++;
        }
        return result;
    }

    Darray GetWeightHistogram(int LayerIdx, int numBins = 50) {
        Darray result(numBins, 0);
        if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return result;
        auto& conv = ConvLayers[LayerIdx];
        std::vector<double> h_weights(conv.weightsSize);
        CUDA_CHECK(cudaMemcpy(h_weights.data(), conv.d_weights, conv.weightsSize * sizeof(double), cudaMemcpyDeviceToHost));
        double minVal = 1e308, maxVal = -1e308;
        for (double v : h_weights) { if (v < minVal) minVal = v; if (v > maxVal) maxVal = v; }
        double range = maxVal - minVal;
        if (range < 1e-8) range = 1;
        for (double v : h_weights) {
            int bin = (int)((v - minVal) / range * (numBins - 1));
            if (bin >= numBins) bin = numBins - 1;
            if (bin < 0) bin = 0;
            result[bin]++;
        }
        return result;
    }

    // Stage 10: Structural Mutations
    void AddFilter(int LayerIdx) {
        if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return;
        auto& conv = ConvLayers[LayerIdx];
        int newNumFilters = conv.numFilters + 1;
        int newWeightsSize = newNumFilters * conv.inputChannels * conv.kernelSize * conv.kernelSize;
        int newOutputSize = newNumFilters * conv.outputHeight * conv.outputWidth;

        // Copy old weights
        std::vector<double> h_weights(conv.weightsSize);
        std::vector<double> h_biases(conv.biasesSize);
        CUDA_CHECK(cudaMemcpy(h_weights.data(), conv.d_weights, conv.weightsSize * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_biases.data(), conv.d_biases, conv.biasesSize * sizeof(double), cudaMemcpyDeviceToHost));

        // Add new filter with random weights
        double scale = std::sqrt(2.0 / (conv.inputChannels * conv.kernelSize * conv.kernelSize));
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        h_weights.resize(newWeightsSize);
        for (int i = conv.weightsSize; i < newWeightsSize; i++) h_weights[i] = dist(rng) * scale;
        h_biases.push_back(0);

        // Reallocate
        cudaFree(conv.d_weights); cudaFree(conv.d_biases);
        cudaFree(conv.d_weightsM); cudaFree(conv.d_weightsV);
        cudaFree(conv.d_biasM); cudaFree(conv.d_biasV);
        cudaFree(conv.d_weightGrads); cudaFree(conv.d_biasGrads);
        cudaFree(conv.d_output); cudaFree(conv.d_preActivation);

        conv.numFilters = newNumFilters;
        conv.weightsSize = newWeightsSize;
        conv.biasesSize = newNumFilters;
        conv.outputSize = newOutputSize;

        CUDA_CHECK(cudaMalloc(&conv.d_weights, conv.weightsSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&conv.d_biases, conv.biasesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&conv.d_weightsM, conv.weightsSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&conv.d_weightsV, conv.weightsSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&conv.d_biasM, conv.biasesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&conv.d_biasV, conv.biasesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&conv.d_weightGrads, conv.weightsSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&conv.d_biasGrads, conv.biasesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&conv.d_output, conv.outputSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&conv.d_preActivation, conv.outputSize * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(conv.d_weights, h_weights.data(), conv.weightsSize * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(conv.d_biases, h_biases.data(), conv.biasesSize * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(conv.d_weightsM, 0, conv.weightsSize * sizeof(double)));
        CUDA_CHECK(cudaMemset(conv.d_weightsV, 0, conv.weightsSize * sizeof(double)));
        CUDA_CHECK(cudaMemset(conv.d_biasM, 0, conv.biasesSize * sizeof(double)));
        CUDA_CHECK(cudaMemset(conv.d_biasV, 0, conv.biasesSize * sizeof(double)));
    }

    int GetNumFilters(int LayerIdx) {
        if (LayerIdx >= 0 && LayerIdx < (int)ConvLayers.size())
            return ConvLayers[LayerIdx].numFilters;
        return 0;
    }

    // Stage 12: Receptive Field
    IntArray GetReceptiveField(int LayerIdx, int Y, int X) {
        IntArray result(4, 0); // startX, endX, startY, endY
        if (LayerIdx < 0 || LayerIdx >= (int)ConvLayers.size()) return result;
        int startX = X, endX = X, startY = Y, endY = Y;
        for (int i = LayerIdx; i >= 0; i--) {
            if (i < (int)PoolLayers.size()) {
                int ps = PoolLayers[i].poolSize;
                startX *= ps; endX = endX * ps + ps - 1;
                startY *= ps; endY = endY * ps + ps - 1;
            }
            int ks = ConvLayers[i].kernelSize;
            int stride = ConvLayers[i].stride;
            int padding = ConvLayers[i].padding;
            startX = startX * stride - padding;
            endX = endX * stride + ks - 1 - padding;
            startY = startY * stride - padding;
            endY = endY * stride + ks - 1 - padding;
        }
        result[0] = startX; result[1] = endX; result[2] = startY; result[3] = endY;
        return result;
    }
};

} // namespace CNNFacade

// ============================================================================
// CLI Main
// ============================================================================

void printHelp(const char* progName) {
    std::cout << "CNNFacade CUDA - GPU-Accelerated CNN Facade\n";
    std::cout << "Matthew Abbott 2025 - Ported from Pascal to C++ to CUDA\n\n";
    std::cout << "Usage: " << progName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help                Show this help message\n";
    std::cout << "  -v, --version             Show version information\n";
    std::cout << "  --device <int>            CUDA device ID (default: 0)\n\n";
    std::cout << "Network Architecture:\n";
    std::cout << "  --input-width <int>       Input image width (default: 28)\n";
    std::cout << "  --input-height <int>      Input image height (default: 28)\n";
    std::cout << "  --input-channels <int>    Input image channels (default: 1)\n";
    std::cout << "  --conv-filters <list>     Comma-separated conv filter counts (default: 32,64)\n";
    std::cout << "  --kernel-sizes <list>     Comma-separated kernel sizes (default: 3,3)\n";
    std::cout << "  --pool-sizes <list>       Comma-separated pool sizes (default: 2,2)\n";
    std::cout << "  --fc-sizes <list>         Comma-separated FC layer sizes (default: 128)\n";
    std::cout << "  --output-size <int>       Number of output classes (default: 10)\n\n";
    std::cout << "Training Parameters:\n";
    std::cout << "  --learning-rate <float>   Learning rate (default: 0.001)\n";
    std::cout << "  --dropout-rate <float>    Dropout rate (default: 0.25)\n";
    std::cout << "  --epochs <int>            Number of training epochs (default: 10)\n";
    std::cout << "  --batch-size <int>        Batch size per epoch (default: 10)\n\n";
    std::cout << "Model I/O:\n";
    std::cout << "  --save <filename>         Save model to file after training\n";
    std::cout << "  --load <filename>         Load model from file before training\n\n";
    std::cout << "Actions:\n";
    std::cout << "  --train                   Run training demo with random data\n";
    std::cout << "  --predict                 Run prediction demo with random data\n";
    std::cout << "  --info                    Display network architecture info\n";
    std::cout << "  --benchmark               Run performance benchmark\n\n";
    std::cout << "Facade Introspection (run after --predict or --train to populate data):\n";
    std::cout << "  --get-feature-map <L,F>        Get feature map for layer L, filter F\n";
    std::cout << "  --get-preactivation <L,F>      Get pre-activation for layer L, filter F\n";
    std::cout << "  --get-kernel <L,F>             Get kernel weights for layer L, filter F\n";
    std::cout << "  --set-bias <L,F,V>             Set bias for layer L, filter F to value V\n";
    std::cout << "  --get-bias <L,F>               Get bias for layer L, filter F\n";
    std::cout << "  --get-filter-gradient <L,F>    Get weight gradients for layer L, filter F\n";
    std::cout << "  --get-bias-gradient <L,F>      Get bias gradient for layer L, filter F\n";
    std::cout << "  --get-pooling-indices <L,F>    Get max pooling indices for layer L, filter F\n";
    std::cout << "  --get-flattened                Get flattened feature vector\n";
    std::cout << "  --get-logits                   Get raw logits from output layer\n";
    std::cout << "  --get-softmax                  Get softmax probabilities\n";
    std::cout << "  --get-layer-stats <L>          Get activation statistics for layer L\n";
    std::cout << "  --get-activation-hist <L>      Get activation histogram for layer L\n";
    std::cout << "  --get-weight-hist <L>          Get weight histogram for layer L\n";
    std::cout << "  --get-receptive-field <L,Y,X>  Get receptive field at layer L, position (Y,X)\n";
    std::cout << "  --get-fc-weights <L,N>         Get FC layer L, neuron N weights\n";
    std::cout << "  --get-fc-bias <L,N>            Get FC layer L, neuron N bias\n";
    std::cout << "  --set-fc-bias <L,N,V>          Set FC layer L, neuron N bias to value V\n";
    std::cout << "  --get-dropout-mask <L>         Get dropout mask for FC layer L\n";
    std::cout << "  --add-filter <L>               Add a new filter to conv layer L\n";
    std::cout << "  --get-num-filters <L>          Get number of filters in layer L\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << progName << " --info\n";
    std::cout << "  " << progName << " --train --epochs 5 --save model.bin\n";
    std::cout << "  " << progName << " --load model.bin --predict\n";
    std::cout << "  " << progName << " --benchmark --batch-size 100\n";
    std::cout << "  " << progName << " --predict --get-feature-map 0,0\n";
    std::cout << "  " << progName << " --predict --get-layer-stats 0 --get-logits\n";
    std::cout << "  " << progName << " --train --get-filter-gradient 0,0\n";
}

std::vector<int> parseIntList(const std::string& s) {
    std::vector<int> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) result.push_back(std::stoi(item));
    return result;
}

std::vector<double> parseDoubleList(const std::string& s) {
    std::vector<double> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) result.push_back(std::stod(item));
    return result;
}

struct FacadeCommand {
    std::string cmd;
    std::string args;
};

void print2DArray(const CNNFacade::D2array& arr, const std::string& name) {
    std::cout << name << " (" << arr.size() << "x" << (arr.empty() ? 0 : arr[0].size()) << "):\n";
    for (size_t h = 0; h < arr.size() && h < 10; h++) {
        std::cout << "  [" << h << "]: ";
        for (size_t w = 0; w < arr[h].size() && w < 10; w++)
            std::cout << std::fixed << std::setprecision(4) << arr[h][w] << " ";
        if (arr[h].size() > 10) std::cout << "...";
        std::cout << "\n";
    }
    if (arr.size() > 10) std::cout << "  ... (" << arr.size() - 10 << " more rows)\n";
}

void print3DArray(const CNNFacade::D3array& arr, const std::string& name) {
    std::cout << name << " (" << arr.size() << "x" << (arr.empty() ? 0 : arr[0].size()) 
              << "x" << (arr.empty() || arr[0].empty() ? 0 : arr[0][0].size()) << "):\n";
    for (size_t c = 0; c < arr.size() && c < 4; c++) {
        std::cout << "  Channel " << c << ":\n";
        for (size_t h = 0; h < arr[c].size() && h < 5; h++) {
            std::cout << "    [" << h << "]: ";
            for (size_t w = 0; w < arr[c][h].size() && w < 5; w++)
                std::cout << std::fixed << std::setprecision(4) << arr[c][h][w] << " ";
            if (arr[c][h].size() > 5) std::cout << "...";
            std::cout << "\n";
        }
        if (arr[c].size() > 5) std::cout << "    ...\n";
    }
    if (arr.size() > 4) std::cout << "  ... (" << arr.size() - 4 << " more channels)\n";
}

void print1DArray(const CNNFacade::Darray& arr, const std::string& name) {
    std::cout << name << " (" << arr.size() << " elements):\n  ";
    for (size_t i = 0; i < arr.size() && i < 20; i++)
        std::cout << std::fixed << std::setprecision(4) << arr[i] << " ";
    if (arr.size() > 20) std::cout << "... (" << arr.size() - 20 << " more)";
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    int inputWidth = 28, inputHeight = 28, inputChannels = 1;
    std::vector<int> convFilters = {32, 64};
    std::vector<int> kernelSizes = {3, 3};
    std::vector<int> poolSizes = {2, 2};
    std::vector<int> fcSizes = {128};
    int outputSize = 10;
    double learningRate = 0.001, dropoutRate = 0.25;
    int epochs = 10, batchSize = 10, deviceId = 0;
    std::string saveFile, loadFile;
    bool doTrain = false, doPredict = false, doInfo = false, doBenchmark = false;
    std::vector<FacadeCommand> facadeCommands;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { printHelp(argv[0]); return 0; }
        else if (arg == "-v" || arg == "--version") { std::cout << "CNNFacade CUDA v1.0.0\n"; return 0; }
        else if (arg == "--device" && i+1 < argc) deviceId = std::stoi(argv[++i]);
        else if (arg == "--input-width" && i+1 < argc) inputWidth = std::stoi(argv[++i]);
        else if (arg == "--input-height" && i+1 < argc) inputHeight = std::stoi(argv[++i]);
        else if (arg == "--input-channels" && i+1 < argc) inputChannels = std::stoi(argv[++i]);
        else if (arg == "--conv-filters" && i+1 < argc) convFilters = parseIntList(argv[++i]);
        else if (arg == "--kernel-sizes" && i+1 < argc) kernelSizes = parseIntList(argv[++i]);
        else if (arg == "--pool-sizes" && i+1 < argc) poolSizes = parseIntList(argv[++i]);
        else if (arg == "--fc-sizes" && i+1 < argc) fcSizes = parseIntList(argv[++i]);
        else if (arg == "--output-size" && i+1 < argc) outputSize = std::stoi(argv[++i]);
        else if (arg == "--learning-rate" && i+1 < argc) learningRate = std::stod(argv[++i]);
        else if (arg == "--dropout-rate" && i+1 < argc) dropoutRate = std::stod(argv[++i]);
        else if (arg == "--epochs" && i+1 < argc) epochs = std::stoi(argv[++i]);
        else if (arg == "--batch-size" && i+1 < argc) batchSize = std::stoi(argv[++i]);
        else if (arg == "--save" && i+1 < argc) saveFile = argv[++i];
        else if (arg == "--load" && i+1 < argc) loadFile = argv[++i];
        else if (arg == "--train") doTrain = true;
        else if (arg == "--predict") doPredict = true;
        else if (arg == "--info") doInfo = true;
        else if (arg == "--benchmark") doBenchmark = true;
        // Facade commands
        else if (arg == "--get-feature-map" && i+1 < argc) facadeCommands.push_back({"get-feature-map", argv[++i]});
        else if (arg == "--get-preactivation" && i+1 < argc) facadeCommands.push_back({"get-preactivation", argv[++i]});
        else if (arg == "--get-kernel" && i+1 < argc) facadeCommands.push_back({"get-kernel", argv[++i]});
        else if (arg == "--set-bias" && i+1 < argc) facadeCommands.push_back({"set-bias", argv[++i]});
        else if (arg == "--get-bias" && i+1 < argc) facadeCommands.push_back({"get-bias", argv[++i]});
        else if (arg == "--get-filter-gradient" && i+1 < argc) facadeCommands.push_back({"get-filter-gradient", argv[++i]});
        else if (arg == "--get-bias-gradient" && i+1 < argc) facadeCommands.push_back({"get-bias-gradient", argv[++i]});
        else if (arg == "--get-pooling-indices" && i+1 < argc) facadeCommands.push_back({"get-pooling-indices", argv[++i]});
        else if (arg == "--get-flattened") facadeCommands.push_back({"get-flattened", ""});
        else if (arg == "--get-logits") facadeCommands.push_back({"get-logits", ""});
        else if (arg == "--get-softmax") facadeCommands.push_back({"get-softmax", ""});
        else if (arg == "--get-layer-stats" && i+1 < argc) facadeCommands.push_back({"get-layer-stats", argv[++i]});
        else if (arg == "--get-activation-hist" && i+1 < argc) facadeCommands.push_back({"get-activation-hist", argv[++i]});
        else if (arg == "--get-weight-hist" && i+1 < argc) facadeCommands.push_back({"get-weight-hist", argv[++i]});
        else if (arg == "--get-receptive-field" && i+1 < argc) facadeCommands.push_back({"get-receptive-field", argv[++i]});
        else if (arg == "--get-fc-weights" && i+1 < argc) facadeCommands.push_back({"get-fc-weights", argv[++i]});
        else if (arg == "--get-fc-bias" && i+1 < argc) facadeCommands.push_back({"get-fc-bias", argv[++i]});
        else if (arg == "--set-fc-bias" && i+1 < argc) facadeCommands.push_back({"set-fc-bias", argv[++i]});
        else if (arg == "--get-dropout-mask" && i+1 < argc) facadeCommands.push_back({"get-dropout-mask", argv[++i]});
        else if (arg == "--add-filter" && i+1 < argc) facadeCommands.push_back({"add-filter", argv[++i]});
        else if (arg == "--get-num-filters" && i+1 < argc) facadeCommands.push_back({"get-num-filters", argv[++i]});
        else { std::cerr << "Unknown option: " << arg << "\nUse --help for usage.\n"; return 1; }
    }

    if (argc == 1) { printHelp(argv[0]); return 0; }

    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(deviceId));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    std::cout << "Using CUDA device " << deviceId << ": " << prop.name << "\n";
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB\n\n";

    while (kernelSizes.size() < convFilters.size()) kernelSizes.push_back(3);
    while (poolSizes.size() < convFilters.size()) poolSizes.push_back(2);

    CNNFacade::TCNNFacade cnn(inputWidth, inputHeight, inputChannels,
                              convFilters, kernelSizes, poolSizes, fcSizes,
                              outputSize, learningRate, dropoutRate);

    if (!loadFile.empty()) {
        std::cout << "Loading model from " << loadFile << "...\n";
        cnn.LoadModel(loadFile);
    }

    if (doInfo) {
        std::cout << "\n=== Network Architecture ===\n";
        std::cout << "Input: " << inputWidth << "x" << inputHeight << "x" << inputChannels << "\n";
        std::cout << "Conv layers: " << cnn.GetNumConvLayers() << "\n";
        for (int i = 0; i < cnn.GetNumConvLayers(); i++) {
            auto cfg = cnn.GetLayerConfig(i);
            std::cout << "  Layer " << i << ": " << cfg.FilterCount << " filters, "
                      << cfg.KernelSize << "x" << cfg.KernelSize << " kernel, "
                      << "stride=" << cfg.Stride << ", padding=" << cfg.Padding
                      << " -> " << cfg.OutputWidth << "x" << cfg.OutputHeight << "\n";
        }
        std::cout << "FC layers: " << cnn.GetNumFCLayers() << "\n";
        for (size_t i = 0; i < fcSizes.size(); i++)
            std::cout << "  FC " << i << ": " << fcSizes[i] << " neurons\n";
        std::cout << "  Output: " << outputSize << " classes\n";
        std::cout << "Total layers: " << cnn.GetNumLayers() << "\n";
        std::cout << "Learning rate: " << learningRate << "\n";
        std::cout << "Dropout rate: " << dropoutRate << "\n";
    }

    if (doTrain) {
        std::cout << "\n=== Training Demo (GPU) ===\n";
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            auto start = std::chrono::high_resolution_clock::now();

            for (int b = 0; b < batchSize; b++) {
                CNNFacade::TImageData img;
                img.Width = inputWidth; img.Height = inputHeight; img.Channels = inputChannels;
                img.Data.resize(inputChannels, CNNFacade::D2array(inputHeight, CNNFacade::Darray(inputWidth)));
                for (int c = 0; c < inputChannels; c++)
                    for (int h = 0; h < inputHeight; h++)
                        for (int w = 0; w < inputWidth; w++)
                            img.Data[c][h][w] = dist(rng);

                CNNFacade::Darray target(outputSize, 0);
                target[rng() % outputSize] = 1.0;

                totalLoss += cnn.TrainStep(img, target);
            }

            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();

            std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                      << " - Loss: " << std::fixed << std::setprecision(6) << (totalLoss / batchSize)
                      << " - Time: " << std::setprecision(1) << ms << " ms"
                      << " (" << std::setprecision(1) << (batchSize * 1000.0 / ms) << " samples/sec)\n";
        }
    }

    if (doBenchmark) {
        std::cout << "\n=== GPU Benchmark ===\n";
        std::mt19937 rng(123);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        // Warmup
        for (int i = 0; i < 10; i++) {
            CNNFacade::TImageData img;
            img.Width = inputWidth; img.Height = inputHeight; img.Channels = inputChannels;
            img.Data.resize(inputChannels, CNNFacade::D2array(inputHeight, CNNFacade::Darray(inputWidth)));
            for (int c = 0; c < inputChannels; c++)
                for (int h = 0; h < inputHeight; h++)
                    for (int w = 0; w < inputWidth; w++)
                        img.Data[c][h][w] = dist(rng);
            cnn.SetTrainingMode(false);
            cnn.Predict(img);
        }

        // Inference benchmark
        int numSamples = 100;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < numSamples; i++) {
            CNNFacade::TImageData img;
            img.Width = inputWidth; img.Height = inputHeight; img.Channels = inputChannels;
            img.Data.resize(inputChannels, CNNFacade::D2array(inputHeight, CNNFacade::Darray(inputWidth)));
            for (int c = 0; c < inputChannels; c++)
                for (int h = 0; h < inputHeight; h++)
                    for (int w = 0; w < inputWidth; w++)
                        img.Data[c][h][w] = dist(rng);
            cnn.Predict(img);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Inference: " << numSamples << " samples in " << std::fixed << std::setprecision(1)
                  << ms << " ms (" << (numSamples * 1000.0 / ms) << " samples/sec)\n";

        // Training benchmark
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < numSamples; i++) {
            CNNFacade::TImageData img;
            img.Width = inputWidth; img.Height = inputHeight; img.Channels = inputChannels;
            img.Data.resize(inputChannels, CNNFacade::D2array(inputHeight, CNNFacade::Darray(inputWidth)));
            for (int c = 0; c < inputChannels; c++)
                for (int h = 0; h < inputHeight; h++)
                    for (int w = 0; w < inputWidth; w++)
                        img.Data[c][h][w] = dist(rng);
            CNNFacade::Darray target(outputSize, 0);
            target[rng() % outputSize] = 1.0;
            cnn.TrainStep(img, target);
        }
        end = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Training:  " << numSamples << " samples in " << std::fixed << std::setprecision(1)
                  << ms << " ms (" << (numSamples * 1000.0 / ms) << " samples/sec)\n";
    }

    if (doPredict) {
        std::cout << "\n=== Prediction Demo ===\n";
        std::mt19937 rng(123);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        CNNFacade::TImageData img;
        img.Width = inputWidth; img.Height = inputHeight; img.Channels = inputChannels;
        img.Data.resize(inputChannels, CNNFacade::D2array(inputHeight, CNNFacade::Darray(inputWidth)));
        for (int c = 0; c < inputChannels; c++)
            for (int h = 0; h < inputHeight; h++)
                for (int w = 0; w < inputWidth; w++)
                    img.Data[c][h][w] = dist(rng);

        cnn.SetTrainingMode(false);
        auto pred = cnn.Predict(img);
        std::cout << "Predictions:\n";
        for (size_t i = 0; i < pred.size(); i++)
            std::cout << "  Class " << i << ": " << std::fixed << std::setprecision(4) << (pred[i] * 100) << "%\n";

        int maxIdx = std::max_element(pred.begin(), pred.end()) - pred.begin();
        std::cout << "Predicted class: " << maxIdx << " (" << std::fixed << std::setprecision(2) << (pred[maxIdx] * 100) << "%)\n";
    }

    if (!saveFile.empty()) {
        std::cout << "\nSaving model to " << saveFile << "...\n";
        cnn.SaveModel(saveFile);
        std::cout << "Model saved.\n";
    }

    // Execute facade commands
    if (!facadeCommands.empty()) {
        std::cout << "\n=== Facade Commands ===\n";
        for (const auto& cmd : facadeCommands) {
            auto params = parseIntList(cmd.args);
            auto dparams = parseDoubleList(cmd.args);
            
            if (cmd.cmd == "get-feature-map" && params.size() >= 2) {
                auto fm = cnn.GetFeatureMap(params[0], params[1]);
                print2DArray(fm, "FeatureMap[" + std::to_string(params[0]) + "][" + std::to_string(params[1]) + "]");
            }
            else if (cmd.cmd == "get-preactivation" && params.size() >= 2) {
                auto pa = cnn.GetPreActivation(params[0], params[1]);
                print2DArray(pa, "PreActivation[" + std::to_string(params[0]) + "][" + std::to_string(params[1]) + "]");
            }
            else if (cmd.cmd == "get-kernel" && params.size() >= 2) {
                auto k = cnn.GetKernel(params[0], params[1]);
                print3DArray(k, "Kernel[" + std::to_string(params[0]) + "][" + std::to_string(params[1]) + "]");
            }
            else if (cmd.cmd == "set-bias" && dparams.size() >= 3) {
                cnn.SetBias((int)dparams[0], (int)dparams[1], dparams[2]);
                std::cout << "Set bias[" << (int)dparams[0] << "][" << (int)dparams[1] << "] = " << dparams[2] << "\n";
            }
            else if (cmd.cmd == "get-bias" && params.size() >= 2) {
                double b = cnn.GetBias(params[0], params[1]);
                std::cout << "Bias[" << params[0] << "][" << params[1] << "] = " << std::fixed << std::setprecision(6) << b << "\n";
            }
            else if (cmd.cmd == "get-filter-gradient" && params.size() >= 2) {
                auto fg = cnn.GetFilterGradient(params[0], params[1]);
                print3DArray(fg, "FilterGradient[" + std::to_string(params[0]) + "][" + std::to_string(params[1]) + "]");
            }
            else if (cmd.cmd == "get-bias-gradient" && params.size() >= 2) {
                double bg = cnn.GetBiasGradient(params[0], params[1]);
                std::cout << "BiasGradient[" << params[0] << "][" << params[1] << "] = " << std::fixed << std::setprecision(6) << bg << "\n";
            }
            else if (cmd.cmd == "get-pooling-indices" && params.size() >= 2) {
                auto pi = cnn.GetPoolingIndices(params[0], params[1]);
                print2DArray(pi, "PoolingIndices[" + std::to_string(params[0]) + "][" + std::to_string(params[1]) + "]");
            }
            else if (cmd.cmd == "get-flattened") {
                auto ff = cnn.GetFlattenedFeatures();
                print1DArray(ff, "FlattenedFeatures");
            }
            else if (cmd.cmd == "get-logits") {
                auto logits = cnn.GetLogits();
                print1DArray(logits, "Logits");
            }
            else if (cmd.cmd == "get-softmax") {
                auto sm = cnn.GetSoftmaxOutput();
                print1DArray(sm, "Softmax");
            }
            else if (cmd.cmd == "get-layer-stats" && params.size() >= 1) {
                auto stats = cnn.GetLayerStats(params[0]);
                std::cout << "LayerStats[" << params[0] << "]:\n";
                std::cout << "  Mean: " << std::fixed << std::setprecision(6) << stats.Mean << "\n";
                std::cout << "  StdDev: " << stats.StdDev << "\n";
                std::cout << "  Min: " << stats.Min << "\n";
                std::cout << "  Max: " << stats.Max << "\n";
                std::cout << "  Count: " << stats.Count << "\n";
            }
            else if (cmd.cmd == "get-activation-hist" && params.size() >= 1) {
                auto hist = cnn.GetActivationHistogram(params[0]);
                print1DArray(hist, "ActivationHistogram[" + std::to_string(params[0]) + "]");
            }
            else if (cmd.cmd == "get-weight-hist" && params.size() >= 1) {
                auto hist = cnn.GetWeightHistogram(params[0]);
                print1DArray(hist, "WeightHistogram[" + std::to_string(params[0]) + "]");
            }
            else if (cmd.cmd == "get-receptive-field" && params.size() >= 3) {
                auto rf = cnn.GetReceptiveField(params[0], params[1], params[2]);
                std::cout << "ReceptiveField[" << params[0] << "] at (" << params[1] << "," << params[2] << "):\n";
                std::cout << "  X: [" << rf[0] << ", " << rf[1] << "]\n";
                std::cout << "  Y: [" << rf[2] << ", " << rf[3] << "]\n";
            }
            else if (cmd.cmd == "get-fc-weights" && params.size() >= 2) {
                auto w = cnn.GetFCWeights(params[0], params[1]);
                print1DArray(w, "FCWeights[" + std::to_string(params[0]) + "][" + std::to_string(params[1]) + "]");
            }
            else if (cmd.cmd == "get-fc-bias" && params.size() >= 2) {
                double b = cnn.GetFCBias(params[0], params[1]);
                std::cout << "FCBias[" << params[0] << "][" << params[1] << "] = " << std::fixed << std::setprecision(6) << b << "\n";
            }
            else if (cmd.cmd == "set-fc-bias" && dparams.size() >= 3) {
                cnn.SetFCBias((int)dparams[0], (int)dparams[1], dparams[2]);
                std::cout << "Set FCBias[" << (int)dparams[0] << "][" << (int)dparams[1] << "] = " << dparams[2] << "\n";
            }
            else if (cmd.cmd == "get-dropout-mask" && params.size() >= 1) {
                auto dm = cnn.GetDropoutMask(params[0]);
                print1DArray(dm, "DropoutMask[" + std::to_string(params[0]) + "]");
            }
            else if (cmd.cmd == "add-filter" && params.size() >= 1) {
                int oldCount = cnn.GetNumFilters(params[0]);
                cnn.AddFilter(params[0]);
                int newCount = cnn.GetNumFilters(params[0]);
                std::cout << "Added filter to layer " << params[0] << ": " << oldCount << " -> " << newCount << " filters\n";
            }
            else if (cmd.cmd == "get-num-filters" && params.size() >= 1) {
                int n = cnn.GetNumFilters(params[0]);
                std::cout << "NumFilters[" << params[0] << "] = " << n << "\n";
            }
            else {
                std::cerr << "Invalid facade command or parameters: " << cmd.cmd << " " << cmd.args << "\n";
            }
        }
    }

    return 0;
}
