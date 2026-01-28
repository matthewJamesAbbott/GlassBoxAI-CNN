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

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <iomanip>
#include <fstream>

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
            exit(1); \
        } \
    } while(0)

// ========== Type Definitions ==========
enum ActivationType {
    atSigmoid,
    atTanh,
    atReLU,
    atLinear
};

enum LossType {
    ltMSE,
    ltCrossEntropy
};

enum PaddingType {
    ptSame,
    ptValid
};

enum Command {
    cmdNone,
    cmdCreate,
    cmdTrain,
    cmdPredict,
    cmdInfo,
    cmdHelp,
    cmdExportONNX,
    cmdImportONNX
};

// Type aliases matching Pascal
typedef vector<double> DArray;
typedef vector<DArray> TDArray2D;
typedef vector<TDArray2D> TDArray3D;
typedef vector<TDArray3D> TDArray4D;
typedef vector<int> TIntArray;

// Batch Normalization Parameters
struct BatchNormParams {
    DArray Gamma;
    DArray Beta;
    DArray RunningMean;
    DArray RunningVar;
    double Epsilon = 1e-5;
    double Momentum = 0.1;
    
    BatchNormParams() : Epsilon(1e-5), Momentum(0.1) {}
    
    void Initialize(int Size) {
        Gamma.resize(Size, 1.0);
        Beta.resize(Size, 0.0);
        RunningMean.resize(Size, 0.0);
        RunningVar.resize(Size, 1.0);
    }
};

struct MaxIndex {
    int X, Y;
};

const double EPSILON = 1e-8;
const double GRAD_CLIP = 1.0;
const int BLOCK_SIZE = 256;
const char MODEL_MAGIC[] = "CNNCUDA1";

// ========== Utility Functions ==========
double ClipValue(double V, double MaxVal) {
    if (V > MaxVal) return MaxVal;
    else if (V < -MaxVal) return -MaxVal;
    else return V;
}

double RandomWeight(double Scale) {
    return (rand() / (double)RAND_MAX - 0.5) * 2.0 * Scale;
}

void InitMatrix(TDArray2D& M, int Rows, int Cols, double Scale) {
    M.resize(Rows);
    for (int i = 0; i < Rows; i++) {
        M[i].resize(Cols);
        for (int j = 0; j < Cols; j++) {
            M[i][j] = RandomWeight(Scale);
        }
    }
}

void ZeroMatrix(TDArray2D& M, int Rows, int Cols) {
    M.resize(Rows);
    for (int i = 0; i < Rows; i++) {
        M[i].resize(Cols);
        for (int j = 0; j < Cols; j++) {
            M[i][j] = 0.0;
        }
    }
}

void ZeroArray(DArray& A, int Size) {
    A.resize(Size);
    for (int i = 0; i < Size; i++) {
        A[i] = 0.0;
    }
}

void Zero3DArray(TDArray3D& A, int D1, int D2, int D3) {
    A.resize(D1);
    for (int i = 0; i < D1; i++) {
        A[i].resize(D2);
        for (int j = 0; j < D2; j++) {
            A[i][j].resize(D3);
            for (int k = 0; k < D3; k++) {
                A[i][j][k] = 0.0;
            }
        }
    }
}

void Zero4DArray(TDArray4D& A, int D1, int D2, int D3, int D4) {
    A.resize(D1);
    for (int i = 0; i < D1; i++) {
        A[i].resize(D2);
        for (int j = 0; j < D2; j++) {
            A[i][j].resize(D3);
            for (int k = 0; k < D3; k++) {
                A[i][j][k].resize(D4);
                for (int l = 0; l < D4; l++) {
                    A[i][j][k][l] = 0.0;
                }
            }
        }
    }
}

// ========== Activation Functions ==========
class TActivation {
public:
    static double Apply(double X, ActivationType ActType) {
        switch (ActType) {
            case atSigmoid:
                return 1.0 / (1.0 + exp(-max(-500.0, min(500.0, X))));
            case atTanh:
                return tanh(X);
            case atReLU:
                return X > 0 ? X : 0;
            case atLinear:
                return X;
            default:
                return X;
        }
    }

    static double Derivative(double Y, ActivationType ActType) {
        switch (ActType) {
            case atSigmoid:
                return Y * (1.0 - Y);
            case atTanh:
                return 1.0 - Y * Y;
            case atReLU:
                return Y > 0 ? 1.0 : 0.0;
            case atLinear:
                return 1.0;
            default:
                return 1.0;
        }
    }

    static void ApplySoftmax(DArray& Arr) {
        double MaxVal = Arr[0];
        for (size_t i = 1; i < Arr.size(); i++) {
            if (Arr[i] > MaxVal) MaxVal = Arr[i];
        }
        double Sum = 0;
        for (size_t i = 0; i < Arr.size(); i++) {
            Arr[i] = exp(Arr[i] - MaxVal);
            Sum += Arr[i];
        }
        for (size_t i = 0; i < Arr.size(); i++) {
            Arr[i] = Arr[i] / Sum;
        }
    }
};

// ========== Loss Functions ==========
class TLoss {
public:
    static double Compute(const DArray& Pred, const DArray& Target, LossType LossType) {
        double Result = 0;
        switch (LossType) {
            case ltMSE:
                for (size_t i = 0; i < Pred.size(); i++) {
                    Result += (Pred[i] - Target[i]) * (Pred[i] - Target[i]);
                }
                break;
            case ltCrossEntropy:
                for (size_t i = 0; i < Pred.size(); i++) {
                    double P = max(1e-15, min(1 - 1e-15, Pred[i]));
                    Result -= (Target[i] * log(P) + (1 - Target[i]) * log(1 - P));
                }
                break;
        }
        return Result / Pred.size();
    }

    static void Gradient(const DArray& Pred, const DArray& Target, LossType LossType, DArray& Grad) {
        Grad.resize(Pred.size());
        switch (LossType) {
            case ltMSE:
                for (size_t i = 0; i < Pred.size(); i++) {
                    Grad[i] = Pred[i] - Target[i];
                }
                break;
            case ltCrossEntropy:
                for (size_t i = 0; i < Pred.size(); i++) {
                    double P = max(1e-15, min(1 - 1e-15, Pred[i]));
                    Grad[i] = (P - Target[i]) / (P * (1 - P) + 1e-15);
                }
                break;
        }
    }
};

// Forward declaration for TConvolutionalNeuralNetworkCUDA
class TConvolutionalNeuralNetworkCUDA;

// Type alias for compatibility with OpenCL version
typedef TConvolutionalNeuralNetworkCUDA TAdvancedCNN;

// ========== Device functions ==========
__device__ double d_ReLU(double x) {
    return (x > 0) ? x : 0.0;
}

__device__ double d_ReLUDerivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

__device__ double d_Clamp(double x, double minVal, double maxVal) {
    if (x < minVal) return minVal;
    if (x > maxVal) return maxVal;
    return x;
}

__device__ double d_ClipGrad(double x) {
    if (!isfinite(x)) return 0;
    return d_Clamp(x, -GRAD_CLIP, GRAD_CLIP);
}

// Custom atomicAdd for double (not natively supported on older architectures)
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

// Kernel: Initialize random states
__global__ void InitRandStates(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Kernel: Conv forward - compute single output pixel
__global__ void ConvForwardKernel(double* output, double* preActivation,
                                   const double* input, const double* weights,
                                   const double* biases,
                                   int inputChannels, int kernelSize,
                                   int inputH, int inputW, int outputH, int outputW,
                                   int stride, int padding, int numFilters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numFilters * outputH * outputW;
    if (idx >= total) return;

    int f = idx / (outputH * outputW);
    int rem = idx % (outputH * outputW);
    int oh = rem / outputW;
    int ow = rem % outputW;

    double sum = biases[f];

    for (int c = 0; c < inputChannels; c++) {
        for (int kh = 0; kh < kernelSize; kh++) {
            for (int kw = 0; kw < kernelSize; kw++) {
                int ih = oh * stride + kh;
                int iw = ow * stride + kw;
                int paddedH = inputH + 2 * padding;
                int paddedW = inputW + 2 * padding;
                int inputIdx = c * paddedH * paddedW + ih * paddedW + iw;
                int weightIdx = f * inputChannels * kernelSize * kernelSize +
                               c * kernelSize * kernelSize + kh * kernelSize + kw;
                sum += input[inputIdx] * weights[weightIdx];
            }
        }
    }

    if (!isfinite(sum)) sum = 0;

    int outIdx = f * outputH * outputW + oh * outputW + ow;
    preActivation[outIdx] = sum;
    output[outIdx] = d_ReLU(sum);
}

// Kernel: Max pooling forward
__global__ void PoolForwardKernel(double* output, int* maxIndicesY, int* maxIndicesX,
                                   const double* input,
                                   int channels, int inputH, int inputW,
                                   int outputH, int outputW, int poolSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * outputH * outputW;
    if (idx >= total) return;

    int c = idx / (outputH * outputW);
    int rem = idx % (outputH * outputW);
    int oh = rem / outputW;
    int ow = rem % outputW;

    double maxVal = -1e308;
    int maxPH = 0, maxPW = 0;

    for (int ph = 0; ph < poolSize; ph++) {
        for (int pw = 0; pw < poolSize; pw++) {
            int ih = oh * poolSize + ph;
            int iw = ow * poolSize + pw;
            int inputIdx = c * inputH * inputW + ih * inputW + iw;
            double val = input[inputIdx];
            if (val > maxVal) {
                maxVal = val;
                maxPH = ph;
                maxPW = pw;
            }
        }
    }

    int outIdx = c * outputH * outputW + oh * outputW + ow;
    output[outIdx] = maxVal;
    maxIndicesY[outIdx] = maxPH;
    maxIndicesX[outIdx] = maxPW;
}

// Kernel: FC forward
__global__ void FCForwardKernel(double* output, double* preActivation,
                                 const double* input, const double* weights,
                                 const double* biases, const double* dropoutMask,
                                 int numNeurons, int numInputs, bool applyReLU) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNeurons) return;

    double sum = biases[i];
    for (int j = 0; j < numInputs; j++) {
        sum += input[j] * weights[i * numInputs + j];
    }

    if (!isfinite(sum)) sum = 0;

    preActivation[i] = sum;
    if (applyReLU)
        output[i] = d_ReLU(sum) * dropoutMask[i];
    else
        output[i] = sum;
}

// Kernel: Softmax compute
__global__ void SoftmaxKernel(double* output, const double* input, int n,
                               double maxVal, double sumExp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double val = exp(input[i] - maxVal) / sumExp;
    if (val < 1e-15) val = 1e-15;
    if (val > 1 - 1e-15) val = 1 - 1e-15;
    output[i] = val;
}

// Kernel: Apply dropout
__global__ void ApplyDropoutKernel(double* dropoutMask, curandState* states,
                                    double dropoutRate, int n, bool isTraining) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (isTraining && dropoutRate > 0) {
        float randVal = curand_uniform(&states[i]);
        if (randVal > dropoutRate)
            dropoutMask[i] = 1.0 / (1.0 - dropoutRate);
        else
            dropoutMask[i] = 0;
    } else {
        dropoutMask[i] = 1.0;
    }
}

// Kernel: FC backward
__global__ void FCBackwardKernel(double* errors, double* inputGrad,
                                  const double* grad, const double* weights,
                                  const double* preActivation, const double* dropoutMask,
                                  int numNeurons, int numInputs, bool isOutputLayer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNeurons) return;

    double delta;
    if (isOutputLayer) {
        delta = grad[i];
    } else {
        delta = grad[i] * d_ReLUDerivative(preActivation[i]) * dropoutMask[i];
    }
    errors[i] = delta;
}

// Kernel: Accumulate input gradients for FC layer
__global__ void FCInputGradKernel(double* inputGrad, const double* errors,
                                   const double* weights, int numNeurons, int numInputs) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= numInputs) return;

    double sum = 0;
    for (int i = 0; i < numNeurons; i++) {
        sum += errors[i] * weights[i * numInputs + j];
    }
    inputGrad[j] = sum;
}

// Kernel: Pool backward
__global__ void PoolBackwardKernel(double* inputGrad, const double* grad,
                                    const int* maxIndicesY, const int* maxIndicesX,
                                    int channels, int inputH, int inputW,
                                    int outputH, int outputW, int poolSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * outputH * outputW;
    if (idx >= total) return;

    int c = idx / (outputH * outputW);
    int rem = idx % (outputH * outputW);
    int oh = rem / outputW;
    int ow = rem % outputW;

    int outIdx = c * outputH * outputW + oh * outputW + ow;
    int srcH = oh * poolSize + maxIndicesY[outIdx];
    int srcW = ow * poolSize + maxIndicesX[outIdx];
    int inputIdx = c * inputH * inputW + srcH * inputW + srcW;

    atomicAddDouble(&inputGrad[inputIdx], grad[outIdx]);
}

// Kernel: Conv backward - compute weight gradients
__global__ void ConvWeightGradKernel(double* weightGrads, double* biasGrads,
                                      const double* gradWithReLU, const double* paddedInput,
                                      int numFilters, int inputChannels, int kernelSize,
                                      int outputH, int outputW, int paddedH, int paddedW,
                                      int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalWeights = numFilters * inputChannels * kernelSize * kernelSize;
    if (idx >= totalWeights) return;

    int f = idx / (inputChannels * kernelSize * kernelSize);
    int rem = idx % (inputChannels * kernelSize * kernelSize);
    int c = rem / (kernelSize * kernelSize);
    rem = rem % (kernelSize * kernelSize);
    int kh = rem / kernelSize;
    int kw = rem % kernelSize;

    double wGrad = 0;
    for (int h = 0; h < outputH; h++) {
        for (int w = 0; w < outputW; w++) {
            int inH = h * stride + kh;
            int inW = w * stride + kw;
            int gradIdx = f * outputH * outputW + h * outputW + w;
            int inputIdx = c * paddedH * paddedW + inH * paddedW + inW;
            wGrad += gradWithReLU[gradIdx] * paddedInput[inputIdx];
        }
    }
    weightGrads[idx] = wGrad;
}

// Kernel: Conv backward - compute bias gradients
__global__ void ConvBiasGradKernel(double* biasGrads, const double* gradWithReLU,
                                    int numFilters, int outputH, int outputW) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= numFilters) return;

    double sum = 0;
    for (int h = 0; h < outputH; h++) {
        for (int w = 0; w < outputW; w++) {
            sum += gradWithReLU[f * outputH * outputW + h * outputW + w];
        }
    }
    biasGrads[f] = sum;
}

// Kernel: Apply ReLU derivative to gradient
__global__ void ApplyReLUDerivKernel(double* gradWithReLU, const double* grad,
                                      const double* preActivation, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    gradWithReLU[i] = grad[i] * d_ReLUDerivative(preActivation[i]);
}

// Kernel: Conv backward - compute input gradients
__global__ void ConvInputGradKernel(double* inputGrad, const double* gradWithReLU,
                                     const double* weights,
                                     int numFilters, int inputChannels, int kernelSize,
                                     int inputH, int inputW, int outputH, int outputW,
                                     int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = inputChannels * inputH * inputW;
    if (idx >= total) return;

    int c = idx / (inputH * inputW);
    int rem = idx % (inputH * inputW);
    int ih = rem / inputW;
    int iw = rem % inputW;

    double sum = 0;
    for (int f = 0; f < numFilters; f++) {
        for (int kh = 0; kh < kernelSize; kh++) {
            for (int kw = 0; kw < kernelSize; kw++) {
                int oh = ih + padding - kh;
                int ow = iw + padding - kw;
                if (oh >= 0 && oh < outputH && ow >= 0 && ow < outputW &&
                    oh % stride == 0 && ow % stride == 0) {
                    oh /= stride;
                    ow /= stride;
                    int gradIdx = f * outputH * outputW + oh * outputW + ow;
                    int weightIdx = f * inputChannels * kernelSize * kernelSize +
                                   c * kernelSize * kernelSize + kh * kernelSize + kw;
                    sum += gradWithReLU[gradIdx] * weights[weightIdx];
                }
            }
        }
    }
    inputGrad[idx] = sum;
}

// Kernel: Adam update for weights
__global__ void AdamUpdateKernel(double* weights, double* M, double* V,
                                  const double* grads, double learningRate,
                                  double beta1, double beta2, int timestep, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double grad = d_ClipGrad(grads[i]);
    M[i] = beta1 * M[i] + (1 - beta1) * grad;
    V[i] = beta2 * V[i] + (1 - beta2) * grad * grad;

    double mHat = M[i] / (1 - pow(beta1, timestep));
    double vHat = V[i] / (1 - pow(beta2, timestep));
    double update = learningRate * mHat / (sqrt(vHat) + EPSILON);

    if (isfinite(update))
        weights[i] -= update;
}

// Kernel: Zero array
__global__ void ZeroArrayKernel(double* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = 0;
}

// Kernel: Pad input
__global__ void PadInputKernel(double* padded, const double* input,
                                int channels, int height, int width, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int paddedH = height + 2 * padding;
    int paddedW = width + 2 * padding;
    int total = channels * paddedH * paddedW;
    if (idx >= total) return;

    int c = idx / (paddedH * paddedW);
    int rem = idx % (paddedH * paddedW);
    int ph = rem / paddedW;
    int pw = rem % paddedW;

    int srcH = ph - padding;
    int srcW = pw - padding;

    if (srcH >= 0 && srcH < height && srcW >= 0 && srcW < width) {
        padded[idx] = input[c * height * width + srcH * width + srcW];
    } else {
        padded[idx] = 0;
    }
}

// Host structures
struct ConvLayerGPU {
    double* d_Weights;
    double* d_Biases;
    double* d_WeightsM;
    double* d_WeightsV;
    double* d_BiasM;
    double* d_BiasV;
    double* d_WeightGrads;
    double* d_BiasGrads;
    double* d_Output;
    double* d_PreActivation;
    double* d_PaddedInput;
    int NumFilters;
    int InputChannels;
    int KernelSize;
    int Stride;
    int Padding;
    int OutputH;
    int OutputW;
};

struct PoolLayerGPU {
    double* d_Output;
    int* d_MaxIndicesY;
    int* d_MaxIndicesX;
    int PoolSize;
    int Stride;
    int OutputH;
    int OutputW;
};

struct FCLayerGPU {
    double* d_Weights;
    double* d_Biases;
    double* d_WeightsM;
    double* d_WeightsV;
    double* d_BiasM;
    double* d_BiasV;
    double* d_Output;
    double* d_PreActivation;
    double* d_Errors;
    double* d_DropoutMask;
    int NumNeurons;
    int NumInputs;
};

class TConvolutionalNeuralNetworkCUDA {
private:
    double LearningRate;
    double DropoutRate;
    double GradientClip;
    double Beta1, Beta2;
    int AdamT;
    bool IsTraining;
    ActivationType HiddenActivation;
    ActivationType OutputActivation;
    LossType LossFunction;

    std::vector<ConvLayerGPU> ConvLayers;
    std::vector<PoolLayerGPU> PoolLayers;
    std::vector<FCLayerGPU> FCLayers;
    FCLayerGPU OutputLayer;

    int InputWidth, InputHeight, InputChannels;
    int FlattenedSize;
    int LastConvH, LastConvW, LastConvC;
    int OutputSize;

    double* d_FlattenedFeatures;
    double* d_InputGrad;
    double* d_ConvGrad;
    double* d_FCGrad;
    double* d_Target;
    double* d_Logits;
    double* d_SoftmaxOutput;

    curandState* d_RandStates;
    int MaxNeurons;

    std::vector<int> FConvFilters;
    std::vector<int> FKernelSizes;
    std::vector<int> FPoolSizes;
    std::vector<int> FFCSizes;

    void AllocateConvLayer(ConvLayerGPU& layer, int numFilters, int inputChannels,
                           int kernelSize, int stride, int padding,
                           int inputH, int inputW);
    void AllocatePoolLayer(PoolLayerGPU& layer, int poolSize, int stride,
                           int inputH, int inputW, int channels);
    void AllocateFCLayer(FCLayerGPU& layer, int numNeurons, int numInputs);
    void FreeConvLayer(ConvLayerGPU& layer);
    void FreePoolLayer(PoolLayerGPU& layer);
    void FreeFCLayer(FCLayerGPU& layer);

public:
    TConvolutionalNeuralNetworkCUDA(int inputWidth, int inputHeight, int inputChannels,
                                    const std::vector<int>& convFilters,
                                    const std::vector<int>& kernelSizes,
                                    const std::vector<int>& poolSizes,
                                    const std::vector<int>& fcSizes,
                                    int outputSize,
                                    ActivationType hiddenAct,
                                    ActivationType outputAct,
                                    LossType lossType,
                                    double learningRate = 0.001,
                                    double gradientClip = 5.0);
    ~TConvolutionalNeuralNetworkCUDA();

    void Predict(const double* imageData, double* result);
    double TrainStep(const double* imageData, const double* target);
    bool Save(const char* filename);
    bool SaveToJSON(const char* filename);
    static TConvolutionalNeuralNetworkCUDA* Load(const char* filename);

    int GetInputWidth() const { return InputWidth; }
    int GetInputHeight() const { return InputHeight; }
    int GetInputChannels() const { return InputChannels; }
    int GetOutputSize() const { return OutputSize; }
    int GetNumConvLayers() const { return ConvLayers.size(); }
    int GetNumFCLayers() const { return FCLayers.size(); }
    double GetLearningRate() const { return LearningRate; }
    double GetGradientClip() const { return GradientClip; }
    ActivationType GetHiddenActivation() const { return HiddenActivation; }
    ActivationType GetOutputActivation() const { return OutputActivation; }
    LossType GetLossType() const { return LossFunction; }
    const std::vector<int>& GetConvFilters() const { return FConvFilters; }
    const std::vector<int>& GetKernelSizes() const { return FKernelSizes; }
    const std::vector<int>& GetPoolSizes() const { return FPoolSizes; }
    const std::vector<int>& GetFCSizes() const { return FFCSizes; }

    // ONNX Export/Import
    void ExportToONNX(const string& Filename);
    static TConvolutionalNeuralNetworkCUDA* ImportFromONNX(const string& Filename);
    
    // Batch Normalization
    bool UseBatchNorm = false;
    vector<BatchNormParams> FBatchNormParams;
    void InitializeBatchNorm();
    DArray ApplyBatchNorm(const DArray& Input, int LayerIdx);
};

void TConvolutionalNeuralNetworkCUDA::AllocateConvLayer(ConvLayerGPU& layer, int numFilters,
                                                        int inputChannels, int kernelSize,
                                                        int stride, int padding,
                                                        int inputH, int inputW) {
    layer.NumFilters = numFilters;
    layer.InputChannels = inputChannels;
    layer.KernelSize = kernelSize;
    layer.Stride = stride;
    layer.Padding = padding;
    layer.OutputH = (inputH + 2 * padding - kernelSize) / stride + 1;
    layer.OutputW = (inputW + 2 * padding - kernelSize) / stride + 1;

    int weightSize = numFilters * inputChannels * kernelSize * kernelSize;
    int outputSize = numFilters * layer.OutputH * layer.OutputW;
    int paddedSize = inputChannels * (inputH + 2 * padding) * (inputW + 2 * padding);

    CUDA_CHECK(cudaMalloc(&layer.d_Weights, weightSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_Biases, numFilters * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_WeightsM, weightSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_WeightsV, weightSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_BiasM, numFilters * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_BiasV, numFilters * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_WeightGrads, weightSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_BiasGrads, numFilters * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_Output, outputSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_PreActivation, outputSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_PaddedInput, paddedSize * sizeof(double)));

    CUDA_CHECK(cudaMemset(layer.d_Biases, 0, numFilters * sizeof(double)));
    CUDA_CHECK(cudaMemset(layer.d_WeightsM, 0, weightSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(layer.d_WeightsV, 0, weightSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(layer.d_BiasM, 0, numFilters * sizeof(double)));
    CUDA_CHECK(cudaMemset(layer.d_BiasV, 0, numFilters * sizeof(double)));

    double scale = sqrt(2.0 / (inputChannels * kernelSize * kernelSize));
    double* h_weights = new double[weightSize];
    for (int i = 0; i < weightSize; i++)
        h_weights[i] = ((double)rand() / RAND_MAX - 0.5) * scale;
    CUDA_CHECK(cudaMemcpy(layer.d_Weights, h_weights, weightSize * sizeof(double), cudaMemcpyHostToDevice));
    delete[] h_weights;
}

void TConvolutionalNeuralNetworkCUDA::AllocatePoolLayer(PoolLayerGPU& layer, int poolSize,
                                                        int stride, int inputH, int inputW,
                                                        int channels) {
    layer.PoolSize = poolSize;
    layer.Stride = stride;
    layer.OutputH = inputH / poolSize;
    layer.OutputW = inputW / poolSize;

    int outputSize = channels * layer.OutputH * layer.OutputW;
    CUDA_CHECK(cudaMalloc(&layer.d_Output, outputSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_MaxIndicesY, outputSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&layer.d_MaxIndicesX, outputSize * sizeof(int)));
}

void TConvolutionalNeuralNetworkCUDA::AllocateFCLayer(FCLayerGPU& layer, int numNeurons,
                                                      int numInputs) {
    layer.NumNeurons = numNeurons;
    layer.NumInputs = numInputs;

    int weightSize = numNeurons * numInputs;
    CUDA_CHECK(cudaMalloc(&layer.d_Weights, weightSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_Biases, numNeurons * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_WeightsM, weightSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_WeightsV, weightSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_BiasM, numNeurons * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_BiasV, numNeurons * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_Output, numNeurons * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_PreActivation, numNeurons * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_Errors, numNeurons * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&layer.d_DropoutMask, numNeurons * sizeof(double)));

    CUDA_CHECK(cudaMemset(layer.d_Biases, 0, numNeurons * sizeof(double)));
    CUDA_CHECK(cudaMemset(layer.d_WeightsM, 0, weightSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(layer.d_WeightsV, 0, weightSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(layer.d_BiasM, 0, numNeurons * sizeof(double)));
    CUDA_CHECK(cudaMemset(layer.d_BiasV, 0, numNeurons * sizeof(double)));

    double scale = sqrt(2.0 / numInputs);
    double* h_weights = new double[weightSize];
    for (int i = 0; i < weightSize; i++)
        h_weights[i] = ((double)rand() / RAND_MAX - 0.5) * scale;
    CUDA_CHECK(cudaMemcpy(layer.d_Weights, h_weights, weightSize * sizeof(double), cudaMemcpyHostToDevice));
    delete[] h_weights;

    double* h_mask = new double[numNeurons];
    for (int i = 0; i < numNeurons; i++) h_mask[i] = 1.0;
    CUDA_CHECK(cudaMemcpy(layer.d_DropoutMask, h_mask, numNeurons * sizeof(double), cudaMemcpyHostToDevice));
    delete[] h_mask;
}

void TConvolutionalNeuralNetworkCUDA::FreeConvLayer(ConvLayerGPU& layer) {
    if (layer.d_Weights) cudaFree(layer.d_Weights);
    if (layer.d_Biases) cudaFree(layer.d_Biases);
    if (layer.d_WeightsM) cudaFree(layer.d_WeightsM);
    if (layer.d_WeightsV) cudaFree(layer.d_WeightsV);
    if (layer.d_BiasM) cudaFree(layer.d_BiasM);
    if (layer.d_BiasV) cudaFree(layer.d_BiasV);
    if (layer.d_WeightGrads) cudaFree(layer.d_WeightGrads);
    if (layer.d_BiasGrads) cudaFree(layer.d_BiasGrads);
    if (layer.d_Output) cudaFree(layer.d_Output);
    if (layer.d_PreActivation) cudaFree(layer.d_PreActivation);
    if (layer.d_PaddedInput) cudaFree(layer.d_PaddedInput);
}

void TConvolutionalNeuralNetworkCUDA::FreePoolLayer(PoolLayerGPU& layer) {
    if (layer.d_Output) cudaFree(layer.d_Output);
    if (layer.d_MaxIndicesY) cudaFree(layer.d_MaxIndicesY);
    if (layer.d_MaxIndicesX) cudaFree(layer.d_MaxIndicesX);
}

void TConvolutionalNeuralNetworkCUDA::FreeFCLayer(FCLayerGPU& layer) {
    if (layer.d_Weights) cudaFree(layer.d_Weights);
    if (layer.d_Biases) cudaFree(layer.d_Biases);
    if (layer.d_WeightsM) cudaFree(layer.d_WeightsM);
    if (layer.d_WeightsV) cudaFree(layer.d_WeightsV);
    if (layer.d_BiasM) cudaFree(layer.d_BiasM);
    if (layer.d_BiasV) cudaFree(layer.d_BiasV);
    if (layer.d_Output) cudaFree(layer.d_Output);
    if (layer.d_PreActivation) cudaFree(layer.d_PreActivation);
    if (layer.d_Errors) cudaFree(layer.d_Errors);
    if (layer.d_DropoutMask) cudaFree(layer.d_DropoutMask);
}

TConvolutionalNeuralNetworkCUDA::TConvolutionalNeuralNetworkCUDA(
    int inputWidth, int inputHeight, int inputChannels,
    const std::vector<int>& convFilters,
    const std::vector<int>& kernelSizes,
    const std::vector<int>& poolSizes,
    const std::vector<int>& fcSizes,
    int outputSize,
    ActivationType hiddenAct,
    ActivationType outputAct,
    LossType lossType,
    double learningRate,
    double gradientClip)
{
    LearningRate = learningRate;
    GradientClip = gradientClip;
    DropoutRate = 0.0;
    HiddenActivation = hiddenAct;
    OutputActivation = outputAct;
    LossFunction = lossType;
    Beta1 = 0.9;
    Beta2 = 0.999;
    AdamT = 0;
    IsTraining = false;

    InputWidth = inputWidth;
    InputHeight = inputHeight;
    InputChannels = inputChannels;
    OutputSize = outputSize;

    FConvFilters = convFilters;
    FKernelSizes = kernelSizes;
    FPoolSizes = poolSizes;
    FFCSizes = fcSizes;

    int currentW = inputWidth;
    int currentH = inputHeight;
    int currentC = inputChannels;

    ConvLayers.resize(convFilters.size());
    PoolLayers.resize(poolSizes.size());

    for (size_t i = 0; i < convFilters.size(); i++) {
        int kernelPadding = kernelSizes[i] / 2;
        AllocateConvLayer(ConvLayers[i], convFilters[i], currentC,
                         kernelSizes[i], 1, kernelPadding, currentH, currentW);
        currentW = ConvLayers[i].OutputW;
        currentH = ConvLayers[i].OutputH;
        currentC = convFilters[i];

        if (i < poolSizes.size()) {
            AllocatePoolLayer(PoolLayers[i], poolSizes[i], poolSizes[i],
                             currentH, currentW, currentC);
            currentW = PoolLayers[i].OutputW;
            currentH = PoolLayers[i].OutputH;
        }
    }

    LastConvH = currentH;
    LastConvW = currentW;
    LastConvC = currentC;
    FlattenedSize = currentW * currentH * currentC;

    FCLayers.resize(fcSizes.size());
    int numInputs = FlattenedSize;

    for (size_t i = 0; i < fcSizes.size(); i++) {
        AllocateFCLayer(FCLayers[i], fcSizes[i], numInputs);
        numInputs = fcSizes[i];
    }

    AllocateFCLayer(OutputLayer, outputSize, numInputs);

    MaxNeurons = FlattenedSize;
    for (size_t i = 0; i < fcSizes.size(); i++)
        if (fcSizes[i] > MaxNeurons) MaxNeurons = fcSizes[i];
    if (outputSize > MaxNeurons) MaxNeurons = outputSize;

    CUDA_CHECK(cudaMalloc(&d_RandStates, MaxNeurons * sizeof(curandState)));
    int blocks = (MaxNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
    InitRandStates<<<blocks, BLOCK_SIZE>>>(d_RandStates, time(nullptr), MaxNeurons);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMalloc(&d_FlattenedFeatures, FlattenedSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_InputGrad, InputChannels * InputHeight * InputWidth * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ConvGrad, FlattenedSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_FCGrad, MaxNeurons * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Target, outputSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Logits, outputSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_SoftmaxOutput, outputSize * sizeof(double)));
}

TConvolutionalNeuralNetworkCUDA::~TConvolutionalNeuralNetworkCUDA() {
    for (auto& layer : ConvLayers) FreeConvLayer(layer);
    for (auto& layer : PoolLayers) FreePoolLayer(layer);
    for (auto& layer : FCLayers) FreeFCLayer(layer);
    FreeFCLayer(OutputLayer);

    cudaFree(d_RandStates);
    cudaFree(d_FlattenedFeatures);
    cudaFree(d_InputGrad);
    cudaFree(d_ConvGrad);
    cudaFree(d_FCGrad);
    cudaFree(d_Target);
    cudaFree(d_Logits);
    cudaFree(d_SoftmaxOutput);
}

void TConvolutionalNeuralNetworkCUDA::Predict(const double* imageData, double* result) {
    IsTraining = false;

    int imageSize = InputChannels * InputHeight * InputWidth;
    double* d_Input;
    CUDA_CHECK(cudaMalloc(&d_Input, imageSize * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_Input, imageData, imageSize * sizeof(double), cudaMemcpyHostToDevice));

    double* currentInput = d_Input;
    int currentH = InputHeight;
    int currentW = InputWidth;
    int currentC = InputChannels;

    for (size_t i = 0; i < ConvLayers.size(); i++) {
        ConvLayerGPU& conv = ConvLayers[i];
        int paddedH = currentH + 2 * conv.Padding;
        int paddedW = currentW + 2 * conv.Padding;
        int paddedSize = currentC * paddedH * paddedW;

        int blocks = (paddedSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        PadInputKernel<<<blocks, BLOCK_SIZE>>>(conv.d_PaddedInput, currentInput,
                                                currentC, currentH, currentW, conv.Padding);

        int outputSize = conv.NumFilters * conv.OutputH * conv.OutputW;
        blocks = (outputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        ConvForwardKernel<<<blocks, BLOCK_SIZE>>>(conv.d_Output, conv.d_PreActivation,
                                                   conv.d_PaddedInput, conv.d_Weights,
                                                   conv.d_Biases, conv.InputChannels,
                                                   conv.KernelSize, currentH, currentW,
                                                   conv.OutputH, conv.OutputW,
                                                   conv.Stride, conv.Padding, conv.NumFilters);

        currentInput = conv.d_Output;
        currentH = conv.OutputH;
        currentW = conv.OutputW;
        currentC = conv.NumFilters;

        if (i < PoolLayers.size()) {
            PoolLayerGPU& pool = PoolLayers[i];
            int poolOutputSize = currentC * pool.OutputH * pool.OutputW;
            blocks = (poolOutputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
            PoolForwardKernel<<<blocks, BLOCK_SIZE>>>(pool.d_Output, pool.d_MaxIndicesY,
                                                       pool.d_MaxIndicesX, currentInput,
                                                       currentC, currentH, currentW,
                                                       pool.OutputH, pool.OutputW, pool.PoolSize);
            currentInput = pool.d_Output;
            currentH = pool.OutputH;
            currentW = pool.OutputW;
        }
    }

    CUDA_CHECK(cudaMemcpy(d_FlattenedFeatures, currentInput, FlattenedSize * sizeof(double), cudaMemcpyDeviceToDevice));

    double* fcInput = d_FlattenedFeatures;
    for (size_t i = 0; i < FCLayers.size(); i++) {
        FCLayerGPU& fc = FCLayers[i];

        int blocks = (fc.NumNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        ApplyDropoutKernel<<<blocks, BLOCK_SIZE>>>(fc.d_DropoutMask, d_RandStates,
                                                    DropoutRate, fc.NumNeurons, IsTraining);
        FCForwardKernel<<<blocks, BLOCK_SIZE>>>(fc.d_Output, fc.d_PreActivation,
                                                 fcInput, fc.d_Weights, fc.d_Biases,
                                                 fc.d_DropoutMask, fc.NumNeurons,
                                                 fc.NumInputs, true);
        fcInput = fc.d_Output;
    }

    int blocks = (OutputLayer.NumNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
    FCForwardKernel<<<blocks, BLOCK_SIZE>>>(d_Logits, OutputLayer.d_PreActivation,
                                             fcInput, OutputLayer.d_Weights,
                                             OutputLayer.d_Biases, OutputLayer.d_DropoutMask,
                                             OutputLayer.NumNeurons, OutputLayer.NumInputs, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    double* h_logits = new double[OutputSize];
    CUDA_CHECK(cudaMemcpy(h_logits, d_Logits, OutputSize * sizeof(double), cudaMemcpyDeviceToHost));

    double maxVal = h_logits[0];
    for (int i = 1; i < OutputSize; i++)
        if (h_logits[i] > maxVal) maxVal = h_logits[i];

    double sumExp = 0;
    for (int i = 0; i < OutputSize; i++)
        sumExp += exp(h_logits[i] - maxVal);

    SoftmaxKernel<<<blocks, BLOCK_SIZE>>>(d_SoftmaxOutput, d_Logits, OutputSize, maxVal, sumExp);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(result, d_SoftmaxOutput, OutputSize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(OutputLayer.d_Output, d_SoftmaxOutput, OutputSize * sizeof(double), cudaMemcpyDeviceToDevice));

    delete[] h_logits;
    cudaFree(d_Input);
}

double TConvolutionalNeuralNetworkCUDA::TrainStep(const double* imageData, const double* target) {
    IsTraining = true;

    double* prediction = new double[OutputSize];
    Predict(imageData, prediction);

    CUDA_CHECK(cudaMemcpy(d_Target, target, OutputSize * sizeof(double), cudaMemcpyHostToDevice));

    double* h_outputGrad = new double[OutputSize];
    for (int i = 0; i < OutputSize; i++)
        h_outputGrad[i] = prediction[i] - target[i];

    CUDA_CHECK(cudaMemcpy(d_FCGrad, h_outputGrad, OutputSize * sizeof(double), cudaMemcpyHostToDevice));

    int blocks = (OutputLayer.NumNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
    FCBackwardKernel<<<blocks, BLOCK_SIZE>>>(OutputLayer.d_Errors, nullptr, d_FCGrad,
                                              OutputLayer.d_Weights, OutputLayer.d_PreActivation,
                                              OutputLayer.d_DropoutMask, OutputLayer.NumNeurons,
                                              OutputLayer.NumInputs, true);

    double* prevOutput = FCLayers.empty() ? d_FlattenedFeatures : FCLayers.back().d_Output;
    int prevSize = FCLayers.empty() ? FlattenedSize : FCLayers.back().NumNeurons;

    CUDA_CHECK(cudaMalloc(&d_FCGrad, prevSize * sizeof(double)));
    blocks = (prevSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    FCInputGradKernel<<<blocks, BLOCK_SIZE>>>(d_FCGrad, OutputLayer.d_Errors,
                                               OutputLayer.d_Weights, OutputLayer.NumNeurons, prevSize);

    for (int i = FCLayers.size() - 1; i >= 0; i--) {
        FCLayerGPU& fc = FCLayers[i];
        blocks = (fc.NumNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        FCBackwardKernel<<<blocks, BLOCK_SIZE>>>(fc.d_Errors, nullptr, d_FCGrad,
                                                 fc.d_Weights, fc.d_PreActivation,
                                                 fc.d_DropoutMask, fc.NumNeurons,
                                                  fc.NumInputs, false);

        prevOutput = (i > 0) ? FCLayers[i - 1].d_Output : d_FlattenedFeatures;
        prevSize = (i > 0) ? FCLayers[i - 1].NumNeurons : FlattenedSize;

        blocks = (prevSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        FCInputGradKernel<<<blocks, BLOCK_SIZE>>>(d_FCGrad, fc.d_Errors,
                                                   fc.d_Weights, fc.NumNeurons, prevSize);
    }

    blocks = (FlattenedSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaMemcpy(d_ConvGrad, d_FCGrad, FlattenedSize * sizeof(double), cudaMemcpyDeviceToDevice));

    double* currentGrad = d_ConvGrad;
    int gradC = LastConvC;

    for (int i = ConvLayers.size() - 1; i >= 0; i--) {
        if (i < (int)PoolLayers.size()) {
            PoolLayerGPU& pool = PoolLayers[i];
            int inputH = ConvLayers[i].OutputH;
            int inputW = ConvLayers[i].OutputW;

            int inputSize = gradC * inputH * inputW;
            double* poolInputGrad;
            CUDA_CHECK(cudaMalloc(&poolInputGrad, inputSize * sizeof(double)));
            blocks = (inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
            ZeroArrayKernel<<<blocks, BLOCK_SIZE>>>(poolInputGrad, inputSize);

            int outputSize = gradC * pool.OutputH * pool.OutputW;
            blocks = (outputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
            PoolBackwardKernel<<<blocks, BLOCK_SIZE>>>(poolInputGrad, currentGrad,
                                                        pool.d_MaxIndicesY, pool.d_MaxIndicesX,
                                                        gradC, inputH, inputW,
                                                        pool.OutputH, pool.OutputW, pool.PoolSize);
            currentGrad = poolInputGrad;
        }

        ConvLayerGPU& conv = ConvLayers[i];

        int outputSize = conv.NumFilters * conv.OutputH * conv.OutputW;
        double* gradWithReLU;
        CUDA_CHECK(cudaMalloc(&gradWithReLU, outputSize * sizeof(double)));
        blocks = (outputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        ApplyReLUDerivKernel<<<blocks, BLOCK_SIZE>>>(gradWithReLU, currentGrad,
                                                      conv.d_PreActivation, outputSize);

        int weightSize = conv.NumFilters * conv.InputChannels * conv.KernelSize * conv.KernelSize;
        int paddedH = (i > 0 ? ConvLayers[i-1].OutputH : InputHeight) + 2 * conv.Padding;
        int paddedW = (i > 0 ? ConvLayers[i-1].OutputW : InputWidth) + 2 * conv.Padding;
        blocks = (weightSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        ConvWeightGradKernel<<<blocks, BLOCK_SIZE>>>(conv.d_WeightGrads, conv.d_BiasGrads,
                                                      gradWithReLU, conv.d_PaddedInput,
                                                      conv.NumFilters, conv.InputChannels,
                                                      conv.KernelSize, conv.OutputH, conv.OutputW,
                                                      paddedH, paddedW, conv.Stride);

        blocks = (conv.NumFilters + BLOCK_SIZE - 1) / BLOCK_SIZE;
        ConvBiasGradKernel<<<blocks, BLOCK_SIZE>>>(conv.d_BiasGrads, gradWithReLU,
                                                    conv.NumFilters, conv.OutputH, conv.OutputW);

        cudaFree(gradWithReLU);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    AdamT++;

    for (auto& conv : ConvLayers) {
        int weightSize = conv.NumFilters * conv.InputChannels * conv.KernelSize * conv.KernelSize;
        blocks = (weightSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        AdamUpdateKernel<<<blocks, BLOCK_SIZE>>>(conv.d_Weights, conv.d_WeightsM, conv.d_WeightsV,
                                                  conv.d_WeightGrads, LearningRate,
                                                  Beta1, Beta2, AdamT, weightSize);
        blocks = (conv.NumFilters + BLOCK_SIZE - 1) / BLOCK_SIZE;
        AdamUpdateKernel<<<blocks, BLOCK_SIZE>>>(conv.d_Biases, conv.d_BiasM, conv.d_BiasV,
                                                  conv.d_BiasGrads, LearningRate,
                                                  Beta1, Beta2, AdamT, conv.NumFilters);
    }

    for (auto& fc : FCLayers) {
        double* d_fcWeightGrads;
        int weightSize = fc.NumNeurons * fc.NumInputs;
        CUDA_CHECK(cudaMalloc(&d_fcWeightGrads, weightSize * sizeof(double)));

        double* h_grads = new double[weightSize];
        double* h_errors = new double[fc.NumNeurons];
        double* h_inputs = new double[fc.NumInputs];
        CUDA_CHECK(cudaMemcpy(h_errors, fc.d_Errors, fc.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));

        double* prevOut = (&fc == &FCLayers[0]) ? d_FlattenedFeatures : FCLayers[&fc - &FCLayers[0] - 1].d_Output;
        CUDA_CHECK(cudaMemcpy(h_inputs, prevOut, fc.NumInputs * sizeof(double), cudaMemcpyDeviceToHost));

        for (int i = 0; i < fc.NumNeurons; i++)
            for (int j = 0; j < fc.NumInputs; j++)
                h_grads[i * fc.NumInputs + j] = h_errors[i] * h_inputs[j];

        CUDA_CHECK(cudaMemcpy(d_fcWeightGrads, h_grads, weightSize * sizeof(double), cudaMemcpyHostToDevice));

        blocks = (weightSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        AdamUpdateKernel<<<blocks, BLOCK_SIZE>>>(fc.d_Weights, fc.d_WeightsM, fc.d_WeightsV,
                                                  d_fcWeightGrads, LearningRate,
                                                  Beta1, Beta2, AdamT, weightSize);
        blocks = (fc.NumNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        AdamUpdateKernel<<<blocks, BLOCK_SIZE>>>(fc.d_Biases, fc.d_BiasM, fc.d_BiasV,
                                                  fc.d_Errors, LearningRate,
                                                  Beta1, Beta2, AdamT, fc.NumNeurons);

        delete[] h_grads;
        delete[] h_errors;
        delete[] h_inputs;
        cudaFree(d_fcWeightGrads);
    }

    {
        double* d_outWeightGrads;
        int weightSize = OutputLayer.NumNeurons * OutputLayer.NumInputs;
        CUDA_CHECK(cudaMalloc(&d_outWeightGrads, weightSize * sizeof(double)));

        double* h_grads = new double[weightSize];
        double* h_errors = new double[OutputLayer.NumNeurons];
        double* h_inputs = new double[OutputLayer.NumInputs];
        CUDA_CHECK(cudaMemcpy(h_errors, OutputLayer.d_Errors, OutputLayer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));

        double* prevOut = FCLayers.empty() ? d_FlattenedFeatures : FCLayers.back().d_Output;
        CUDA_CHECK(cudaMemcpy(h_inputs, prevOut, OutputLayer.NumInputs * sizeof(double), cudaMemcpyDeviceToHost));

        for (int i = 0; i < OutputLayer.NumNeurons; i++)
            for (int j = 0; j < OutputLayer.NumInputs; j++)
                h_grads[i * OutputLayer.NumInputs + j] = h_errors[i] * h_inputs[j];

        CUDA_CHECK(cudaMemcpy(d_outWeightGrads, h_grads, weightSize * sizeof(double), cudaMemcpyHostToDevice));

        blocks = (weightSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        AdamUpdateKernel<<<blocks, BLOCK_SIZE>>>(OutputLayer.d_Weights, OutputLayer.d_WeightsM, OutputLayer.d_WeightsV,
                                                  d_outWeightGrads, LearningRate,
                                                  Beta1, Beta2, AdamT, weightSize);
        blocks = (OutputLayer.NumNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        AdamUpdateKernel<<<blocks, BLOCK_SIZE>>>(OutputLayer.d_Biases, OutputLayer.d_BiasM, OutputLayer.d_BiasV,
                                                  OutputLayer.d_Errors, LearningRate,
                                                  Beta1, Beta2, AdamT, OutputLayer.NumNeurons);

        delete[] h_grads;
        delete[] h_errors;
        delete[] h_inputs;
        cudaFree(d_outWeightGrads);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    double loss = 0;
    for (int i = 0; i < OutputSize; i++) {
        if (target[i] > 0) {
            double p = prediction[i];
            if (p < 1e-15) p = 1e-15;
            if (p > 1 - 1e-15) p = 1 - 1e-15;
            loss -= target[i] * log(p);
        }
    }

    delete[] prediction;
    delete[] h_outputGrad;

    return loss;
}

bool TConvolutionalNeuralNetworkCUDA::Save(const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return false;

    fwrite(MODEL_MAGIC, 1, 8, f);
    fwrite(&InputWidth, sizeof(int), 1, f);
    fwrite(&InputHeight, sizeof(int), 1, f);
    fwrite(&InputChannels, sizeof(int), 1, f);
    fwrite(&OutputSize, sizeof(int), 1, f);
    fwrite(&LearningRate, sizeof(double), 1, f);
    fwrite(&DropoutRate, sizeof(double), 1, f);
    fwrite(&Beta1, sizeof(double), 1, f);
    fwrite(&Beta2, sizeof(double), 1, f);
    fwrite(&AdamT, sizeof(int), 1, f);
    fwrite(&FlattenedSize, sizeof(int), 1, f);
    fwrite(&LastConvH, sizeof(int), 1, f);
    fwrite(&LastConvW, sizeof(int), 1, f);
    fwrite(&LastConvC, sizeof(int), 1, f);

    int numConv = FConvFilters.size();
    fwrite(&numConv, sizeof(int), 1, f);
    fwrite(FConvFilters.data(), sizeof(int), numConv, f);
    fwrite(FKernelSizes.data(), sizeof(int), numConv, f);

    int numPool = FPoolSizes.size();
    fwrite(&numPool, sizeof(int), 1, f);
    fwrite(FPoolSizes.data(), sizeof(int), numPool, f);

    int numFC = FFCSizes.size();
    fwrite(&numFC, sizeof(int), 1, f);
    fwrite(FFCSizes.data(), sizeof(int), numFC, f);

    for (auto& conv : ConvLayers) {
        int weightSize = conv.NumFilters * conv.InputChannels * conv.KernelSize * conv.KernelSize;
        double* h_data = new double[weightSize];

        CUDA_CHECK(cudaMemcpy(h_data, conv.d_Weights, weightSize * sizeof(double), cudaMemcpyDeviceToHost));
        fwrite(h_data, sizeof(double), weightSize, f);
        CUDA_CHECK(cudaMemcpy(h_data, conv.d_WeightsM, weightSize * sizeof(double), cudaMemcpyDeviceToHost));
        fwrite(h_data, sizeof(double), weightSize, f);
        CUDA_CHECK(cudaMemcpy(h_data, conv.d_WeightsV, weightSize * sizeof(double), cudaMemcpyDeviceToHost));
        fwrite(h_data, sizeof(double), weightSize, f);

        delete[] h_data;

        double* h_bias = new double[conv.NumFilters];
        CUDA_CHECK(cudaMemcpy(h_bias, conv.d_Biases, conv.NumFilters * sizeof(double), cudaMemcpyDeviceToHost));
        fwrite(h_bias, sizeof(double), conv.NumFilters, f);
        CUDA_CHECK(cudaMemcpy(h_bias, conv.d_BiasM, conv.NumFilters * sizeof(double), cudaMemcpyDeviceToHost));
        fwrite(h_bias, sizeof(double), conv.NumFilters, f);
        CUDA_CHECK(cudaMemcpy(h_bias, conv.d_BiasV, conv.NumFilters * sizeof(double), cudaMemcpyDeviceToHost));
        fwrite(h_bias, sizeof(double), conv.NumFilters, f);

        delete[] h_bias;
    }

    auto saveFCLayer = [&](FCLayerGPU& layer) {
        int weightSize = layer.NumNeurons * layer.NumInputs;
        double* h_weights = new double[weightSize];
        double* h_biases = new double[layer.NumNeurons];

        CUDA_CHECK(cudaMemcpy(h_weights, layer.d_Weights, weightSize * sizeof(double), cudaMemcpyDeviceToHost));
        fwrite(h_weights, sizeof(double), weightSize, f);
        CUDA_CHECK(cudaMemcpy(h_weights, layer.d_WeightsM, weightSize * sizeof(double), cudaMemcpyDeviceToHost));
        fwrite(h_weights, sizeof(double), weightSize, f);
        CUDA_CHECK(cudaMemcpy(h_weights, layer.d_WeightsV, weightSize * sizeof(double), cudaMemcpyDeviceToHost));
        fwrite(h_weights, sizeof(double), weightSize, f);

        CUDA_CHECK(cudaMemcpy(h_biases, layer.d_Biases, layer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));
        fwrite(h_biases, sizeof(double), layer.NumNeurons, f);
        CUDA_CHECK(cudaMemcpy(h_biases, layer.d_BiasM, layer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));
        fwrite(h_biases, sizeof(double), layer.NumNeurons, f);
        CUDA_CHECK(cudaMemcpy(h_biases, layer.d_BiasV, layer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));
        fwrite(h_biases, sizeof(double), layer.NumNeurons, f);

        delete[] h_weights;
        delete[] h_biases;
    };

    for (auto& fc : FCLayers)
        saveFCLayer(fc);
    saveFCLayer(OutputLayer);

    fclose(f);
    return true;
}

bool TConvolutionalNeuralNetworkCUDA::SaveToJSON(const char* filename) {
    ofstream outFile(filename);
    if (!outFile.is_open()) return false;
    
    auto ActivationToString = [](ActivationType a) -> string {
        switch(a) {
            case atReLU: return "relu";
            case atSigmoid: return "sigmoid";
            case atTanh: return "tanh";
            case atLinear: return "linear";
            default: return "relu";
        }
    };
    
    auto LossToString = [](LossType l) -> string {
        switch(l) {
            case ltMSE: return "mse";
            case ltCrossEntropy: return "crossentropy";
            default: return "mse";
        }
    };
    
    outFile << "{\n";
    outFile << "  \"input_width\": " << InputWidth << ",\n";
    outFile << "  \"input_height\": " << InputHeight << ",\n";
    outFile << "  \"input_channels\": " << InputChannels << ",\n";
    outFile << "  \"output_size\": " << OutputSize << ",\n";
    
    outFile << "  \"conv_filters\": [";
    for (size_t i = 0; i < FConvFilters.size(); i++) {
        if (i > 0) outFile << ", ";
        outFile << FConvFilters[i];
    }
    outFile << "],\n";
    
    outFile << "  \"kernel_sizes\": [";
    for (size_t i = 0; i < FKernelSizes.size(); i++) {
        if (i > 0) outFile << ", ";
        outFile << FKernelSizes[i];
    }
    outFile << "],\n";
    
    outFile << "  \"pool_sizes\": [";
    for (size_t i = 0; i < FPoolSizes.size(); i++) {
        if (i > 0) outFile << ", ";
        outFile << FPoolSizes[i];
    }
    outFile << "],\n";
    
    outFile << "  \"fc_layer_sizes\": [";
    for (size_t i = 0; i < FFCSizes.size(); i++) {
        if (i > 0) outFile << ", ";
        outFile << FFCSizes[i];
    }
    outFile << "],\n";
    
    outFile << fixed << setprecision(6);
    outFile << "  \"learning_rate\": " << LearningRate << ",\n";
    outFile << "  \"dropout_rate\": " << DropoutRate << ",\n";
    outFile << "  \"activation\": \"" << ActivationToString(HiddenActivation) << "\",\n";
    outFile << "  \"output_activation\": \"" << ActivationToString(OutputActivation) << "\",\n";
    outFile << "  \"loss_type\": \"" << LossToString(LossFunction) << "\",\n";
    
    outFile << "  \"conv_layers\": [\n";
    for (size_t i = 0; i < ConvLayers.size(); i++) {
        auto& conv = ConvLayers[i];
        int weightSize = conv.NumFilters * conv.InputChannels * conv.KernelSize * conv.KernelSize;
        double* h_weights = new double[weightSize];
        double* h_biases = new double[conv.NumFilters];
        
        CUDA_CHECK(cudaMemcpy(h_weights, conv.d_Weights, weightSize * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_biases, conv.d_Biases, conv.NumFilters * sizeof(double), cudaMemcpyDeviceToHost));
        
        outFile << "    {\n";
        outFile << "      \"filters\": [\n";
        for (int f = 0; f < conv.NumFilters; f++) {
            outFile << "        {\n";
            outFile << "          \"bias\": " << h_biases[f] << ",\n";
            outFile << "          \"weights\": [";
            int filterOffset = f * conv.InputChannels * conv.KernelSize * conv.KernelSize;
            for (int c = 0; c < conv.InputChannels; c++) {
                if (c > 0) outFile << ", ";
                outFile << "[";
                for (int ky = 0; ky < conv.KernelSize; ky++) {
                    if (ky > 0) outFile << ", ";
                    outFile << "[";
                    for (int kx = 0; kx < conv.KernelSize; kx++) {
                        if (kx > 0) outFile << ", ";
                        int idx = filterOffset + c * conv.KernelSize * conv.KernelSize + ky * conv.KernelSize + kx;
                        outFile << h_weights[idx];
                    }
                    outFile << "]";
                }
                outFile << "]";
            }
            outFile << "]\n";
            outFile << "        }" << (f < conv.NumFilters - 1 ? "," : "") << "\n";
        }
        outFile << "      ]\n";
        outFile << "    }" << (i < ConvLayers.size() - 1 ? "," : "") << "\n";
        
        delete[] h_weights;
        delete[] h_biases;
    }
    outFile << "  ],\n";
    
    outFile << "  \"pool_layers\": [\n";
    for (size_t i = 0; i < PoolLayers.size(); i++) {
        outFile << "    {\"poolSize\": " << PoolLayers[i].PoolSize << "}";
        outFile << (i < PoolLayers.size() - 1 ? "," : "") << "\n";
    }
    outFile << "  ],\n";
    
    outFile << "  \"fc_layers\": [\n";
    for (size_t i = 0; i < FCLayers.size(); i++) {
        auto& fc = FCLayers[i];
        int weightSize = fc.NumNeurons * fc.NumInputs;
        double* h_weights = new double[weightSize];
        double* h_biases = new double[fc.NumNeurons];
        
        CUDA_CHECK(cudaMemcpy(h_weights, fc.d_Weights, weightSize * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_biases, fc.d_Biases, fc.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));
        
        outFile << "    {\n";
        outFile << "      \"neurons\": [\n";
        for (int j = 0; j < fc.NumNeurons; j++) {
            outFile << "        {\n";
            outFile << "          \"bias\": " << h_biases[j] << ",\n";
            outFile << "          \"weights\": [";
            for (int w = 0; w < fc.NumInputs; w++) {
                if (w > 0) outFile << ", ";
                outFile << h_weights[j * fc.NumInputs + w];
            }
            outFile << "]\n";
            outFile << "        }" << (j < fc.NumNeurons - 1 ? "," : "") << "\n";
        }
        outFile << "      ]\n";
        outFile << "    }" << (i < FCLayers.size() - 1 ? "," : "") << "\n";
        
        delete[] h_weights;
        delete[] h_biases;
    }
    outFile << "  ],\n";
    
    outFile << "  \"output_layer\": {\n";
    outFile << "    \"neurons\": [\n";
    {
        int weightSize = OutputLayer.NumNeurons * OutputLayer.NumInputs;
        double* h_weights = new double[weightSize];
        double* h_biases = new double[OutputLayer.NumNeurons];
        
        CUDA_CHECK(cudaMemcpy(h_weights, OutputLayer.d_Weights, weightSize * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_biases, OutputLayer.d_Biases, OutputLayer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));
        
        for (int j = 0; j < OutputLayer.NumNeurons; j++) {
            outFile << "      {\n";
            outFile << "        \"bias\": " << h_biases[j] << ",\n";
            outFile << "        \"weights\": [";
            for (int w = 0; w < OutputLayer.NumInputs; w++) {
                if (w > 0) outFile << ", ";
                outFile << h_weights[j * OutputLayer.NumInputs + w];
            }
            outFile << "]\n";
            outFile << "      }" << (j < OutputLayer.NumNeurons - 1 ? "," : "") << "\n";
        }
        
        delete[] h_weights;
        delete[] h_biases;
    }
    outFile << "    ]\n";
    outFile << "  }\n";
    outFile << "}\n";
    
    outFile.close();
    return true;
}

TConvolutionalNeuralNetworkCUDA* TConvolutionalNeuralNetworkCUDA::Load(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return nullptr;

    char magic[9] = {0};
    fread(magic, 1, 8, f);
    if (strcmp(magic, MODEL_MAGIC) != 0) {
        fclose(f);
        return nullptr;
    }

    int inputWidth, inputHeight, inputChannels, outputSize;
    double learningRate, dropoutRate, beta1, beta2;
    int adamT, flattenedSize, lastConvH, lastConvW, lastConvC;

    fread(&inputWidth, sizeof(int), 1, f);
    fread(&inputHeight, sizeof(int), 1, f);
    fread(&inputChannels, sizeof(int), 1, f);
    fread(&outputSize, sizeof(int), 1, f);
    fread(&learningRate, sizeof(double), 1, f);
    fread(&dropoutRate, sizeof(double), 1, f);
    fread(&beta1, sizeof(double), 1, f);
    fread(&beta2, sizeof(double), 1, f);
    fread(&adamT, sizeof(int), 1, f);
    fread(&flattenedSize, sizeof(int), 1, f);
    fread(&lastConvH, sizeof(int), 1, f);
    fread(&lastConvW, sizeof(int), 1, f);
    fread(&lastConvC, sizeof(int), 1, f);

    int numConv;
    fread(&numConv, sizeof(int), 1, f);
    std::vector<int> convFilters(numConv), kernelSizes(numConv);
    fread(convFilters.data(), sizeof(int), numConv, f);
    fread(kernelSizes.data(), sizeof(int), numConv, f);

    int numPool;
    fread(&numPool, sizeof(int), 1, f);
    std::vector<int> poolSizes(numPool);
    fread(poolSizes.data(), sizeof(int), numPool, f);

    int numFC;
    fread(&numFC, sizeof(int), 1, f);
    std::vector<int> fcSizes(numFC);
    fread(fcSizes.data(), sizeof(int), numFC, f);

    TConvolutionalNeuralNetworkCUDA* cnn = new TConvolutionalNeuralNetworkCUDA(
        inputWidth, inputHeight, inputChannels,
        convFilters, kernelSizes, poolSizes, fcSizes, outputSize,
        atReLU, atLinear, ltMSE,
        learningRate, 5.0);

    cnn->Beta1 = beta1;
    cnn->Beta2 = beta2;
    cnn->AdamT = adamT;

    for (auto& conv : cnn->ConvLayers) {
        int weightSize = conv.NumFilters * conv.InputChannels * conv.KernelSize * conv.KernelSize;
        double* h_data = new double[weightSize];

        fread(h_data, sizeof(double), weightSize, f);
        CUDA_CHECK(cudaMemcpy(conv.d_Weights, h_data, weightSize * sizeof(double), cudaMemcpyHostToDevice));
        fread(h_data, sizeof(double), weightSize, f);
        CUDA_CHECK(cudaMemcpy(conv.d_WeightsM, h_data, weightSize * sizeof(double), cudaMemcpyHostToDevice));
        fread(h_data, sizeof(double), weightSize, f);
        CUDA_CHECK(cudaMemcpy(conv.d_WeightsV, h_data, weightSize * sizeof(double), cudaMemcpyHostToDevice));

        delete[] h_data;

        double* h_bias = new double[conv.NumFilters];
        fread(h_bias, sizeof(double), conv.NumFilters, f);
        CUDA_CHECK(cudaMemcpy(conv.d_Biases, h_bias, conv.NumFilters * sizeof(double), cudaMemcpyHostToDevice));
        fread(h_bias, sizeof(double), conv.NumFilters, f);
        CUDA_CHECK(cudaMemcpy(conv.d_BiasM, h_bias, conv.NumFilters * sizeof(double), cudaMemcpyHostToDevice));
        fread(h_bias, sizeof(double), conv.NumFilters, f);
        CUDA_CHECK(cudaMemcpy(conv.d_BiasV, h_bias, conv.NumFilters * sizeof(double), cudaMemcpyHostToDevice));

        delete[] h_bias;
    }

    auto loadFCLayer = [&](FCLayerGPU& layer) {
        int weightSize = layer.NumNeurons * layer.NumInputs;
        double* h_weights = new double[weightSize];
        double* h_biases = new double[layer.NumNeurons];

        fread(h_weights, sizeof(double), weightSize, f);
        CUDA_CHECK(cudaMemcpy(layer.d_Weights, h_weights, weightSize * sizeof(double), cudaMemcpyHostToDevice));
        fread(h_weights, sizeof(double), weightSize, f);
        CUDA_CHECK(cudaMemcpy(layer.d_WeightsM, h_weights, weightSize * sizeof(double), cudaMemcpyHostToDevice));
        fread(h_weights, sizeof(double), weightSize, f);
        CUDA_CHECK(cudaMemcpy(layer.d_WeightsV, h_weights, weightSize * sizeof(double), cudaMemcpyHostToDevice));

        fread(h_biases, sizeof(double), layer.NumNeurons, f);
        CUDA_CHECK(cudaMemcpy(layer.d_Biases, h_biases, layer.NumNeurons * sizeof(double), cudaMemcpyHostToDevice));
        fread(h_biases, sizeof(double), layer.NumNeurons, f);
        CUDA_CHECK(cudaMemcpy(layer.d_BiasM, h_biases, layer.NumNeurons * sizeof(double), cudaMemcpyHostToDevice));
        fread(h_biases, sizeof(double), layer.NumNeurons, f);
        CUDA_CHECK(cudaMemcpy(layer.d_BiasV, h_biases, layer.NumNeurons * sizeof(double), cudaMemcpyHostToDevice));

        delete[] h_weights;
        delete[] h_biases;
    };

    for (auto& fc : cnn->FCLayers)
        loadFCLayer(fc);
    loadFCLayer(cnn->OutputLayer);

    fclose(f);
    return cnn;
}

// ========== Batch Normalization Methods ==========
void TConvolutionalNeuralNetworkCUDA::InitializeBatchNorm() {
    FBatchNormParams.clear();
    for (size_t i = 0; i < ConvLayers.size(); i++) {
        BatchNormParams params;
        params.Initialize(FConvFilters[i]);
        FBatchNormParams.push_back(params);
    }
}

DArray TConvolutionalNeuralNetworkCUDA::ApplyBatchNorm(const DArray& Input, int LayerIdx) {
    if (!UseBatchNorm || LayerIdx >= (int)FBatchNormParams.size()) {
        return Input;
    }
    
    const BatchNormParams& params = FBatchNormParams[LayerIdx];
    DArray Output(Input.size());
    
    int channelSize = Input.size() / params.Gamma.size();
    
    for (size_t c = 0; c < params.Gamma.size(); c++) {
        for (int i = 0; i < channelSize; i++) {
            int idx = c * channelSize + i;
            if (idx < (int)Input.size()) {
                double normalized = (Input[idx] - params.RunningMean[c]) / 
                                   sqrt(params.RunningVar[c] + params.Epsilon);
                Output[idx] = params.Gamma[c] * normalized + params.Beta[c];
            }
        }
    }
    
    return Output;
}

// ========== ONNX Export/Import Methods ==========
void TConvolutionalNeuralNetworkCUDA::ExportToONNX(const string& Filename) {
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
    file.write(reinterpret_cast<const char*>(&InputWidth), sizeof(int));
    file.write(reinterpret_cast<const char*>(&InputHeight), sizeof(int));
    file.write(reinterpret_cast<const char*>(&InputChannels), sizeof(int));
    file.write(reinterpret_cast<const char*>(&OutputSize), sizeof(int));
    
    int useBN = UseBatchNorm ? 1 : 0;
    file.write(reinterpret_cast<const char*>(&useBN), sizeof(int));
    
    // Write conv layer count and metadata
    int numConvLayers = ConvLayers.size();
    file.write(reinterpret_cast<const char*>(&numConvLayers), sizeof(int));
    
    for (int i = 0; i < numConvLayers; i++) {
        file.write(reinterpret_cast<const char*>(&FConvFilters[i]), sizeof(int));
        file.write(reinterpret_cast<const char*>(&FKernelSizes[i]), sizeof(int));
        file.write(reinterpret_cast<const char*>(&FPoolSizes[i]), sizeof(int));
    }
    
    // Write FC layer count and sizes
    int numFCLayers = FFCSizes.size();
    file.write(reinterpret_cast<const char*>(&numFCLayers), sizeof(int));
    for (int i = 0; i < numFCLayers; i++) {
        file.write(reinterpret_cast<const char*>(&FFCSizes[i]), sizeof(int));
    }
    
    // Write conv layer weights (copy from GPU to host first)
    for (size_t i = 0; i < ConvLayers.size(); i++) {
        ConvLayerGPU& conv = ConvLayers[i];
        int weightSize = conv.NumFilters * conv.InputChannels * conv.KernelSize * conv.KernelSize;
        
        double* h_weights = new double[weightSize];
        double* h_biases = new double[conv.NumFilters];
        
        CUDA_CHECK(cudaMemcpy(h_weights, conv.d_Weights, weightSize * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_biases, conv.d_Biases, conv.NumFilters * sizeof(double), cudaMemcpyDeviceToHost));
        
        int numFilters = conv.NumFilters;
        file.write(reinterpret_cast<const char*>(&numFilters), sizeof(int));
        
        for (int f = 0; f < numFilters; f++) {
            // Write filter dimensions
            int d0 = conv.InputChannels;
            int d1 = 1;
            int d2 = conv.KernelSize;
            int d3 = conv.KernelSize;
            
            file.write(reinterpret_cast<const char*>(&d0), sizeof(int));
            file.write(reinterpret_cast<const char*>(&d1), sizeof(int));
            file.write(reinterpret_cast<const char*>(&d2), sizeof(int));
            file.write(reinterpret_cast<const char*>(&d3), sizeof(int));
            
            // Write weights for this filter
            int filterOffset = f * conv.InputChannels * conv.KernelSize * conv.KernelSize;
            for (int c = 0; c < conv.InputChannels; c++) {
                for (int ky = 0; ky < conv.KernelSize; ky++) {
                    for (int kx = 0; kx < conv.KernelSize; kx++) {
                        int idx = filterOffset + c * conv.KernelSize * conv.KernelSize + ky * conv.KernelSize + kx;
                        file.write(reinterpret_cast<const char*>(&h_weights[idx]), sizeof(double));
                    }
                }
            }
            
            // Write bias
            file.write(reinterpret_cast<const char*>(&h_biases[f]), sizeof(double));
        }
        
        delete[] h_weights;
        delete[] h_biases;
    }
    
    // Write FC layer weights
    for (size_t i = 0; i < FCLayers.size(); i++) {
        FCLayerGPU& fc = FCLayers[i];
        int weightSize = fc.NumNeurons * fc.NumInputs;
        
        double* h_weights = new double[weightSize];
        double* h_biases = new double[fc.NumNeurons];
        
        CUDA_CHECK(cudaMemcpy(h_weights, fc.d_Weights, weightSize * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_biases, fc.d_Biases, fc.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));
        
        int rows = fc.NumNeurons;
        int cols = fc.NumInputs;
        
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                file.write(reinterpret_cast<const char*>(&h_weights[r * cols + c]), sizeof(double));
            }
        }
        
        int biasSize = fc.NumNeurons;
        file.write(reinterpret_cast<const char*>(&biasSize), sizeof(int));
        for (int b = 0; b < biasSize; b++) {
            file.write(reinterpret_cast<const char*>(&h_biases[b]), sizeof(double));
        }
        
        delete[] h_weights;
        delete[] h_biases;
    }
    
    // Write output layer
    {
        int weightSize = OutputLayer.NumNeurons * OutputLayer.NumInputs;
        double* h_weights = new double[weightSize];
        double* h_biases = new double[OutputLayer.NumNeurons];
        
        CUDA_CHECK(cudaMemcpy(h_weights, OutputLayer.d_Weights, weightSize * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_biases, OutputLayer.d_Biases, OutputLayer.NumNeurons * sizeof(double), cudaMemcpyDeviceToHost));
        
        int outRows = OutputLayer.NumNeurons;
        int outCols = OutputLayer.NumInputs;
        file.write(reinterpret_cast<const char*>(&outRows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&outCols), sizeof(int));
        
        for (int r = 0; r < outRows; r++) {
            for (int c = 0; c < outCols; c++) {
                file.write(reinterpret_cast<const char*>(&h_weights[r * outCols + c]), sizeof(double));
            }
        }
        
        int outBiasSize = OutputLayer.NumNeurons;
        file.write(reinterpret_cast<const char*>(&outBiasSize), sizeof(int));
        for (int b = 0; b < outBiasSize; b++) {
            file.write(reinterpret_cast<const char*>(&h_biases[b]), sizeof(double));
        }
        
        delete[] h_weights;
        delete[] h_biases;
    }
    
    // Write batch norm params if enabled
    if (UseBatchNorm) {
        for (size_t i = 0; i < FBatchNormParams.size(); i++) {
            const BatchNormParams& bn = FBatchNormParams[i];
            int size = bn.Gamma.size();
            file.write(reinterpret_cast<const char*>(&size), sizeof(int));
            
            for (int j = 0; j < size; j++) {
                file.write(reinterpret_cast<const char*>(&bn.Gamma[j]), sizeof(double));
                file.write(reinterpret_cast<const char*>(&bn.Beta[j]), sizeof(double));
                file.write(reinterpret_cast<const char*>(&bn.RunningMean[j]), sizeof(double));
                file.write(reinterpret_cast<const char*>(&bn.RunningVar[j]), sizeof(double));
            }
        }
    }
    
    file.close();
    cout << "Model exported to ONNX: " << Filename << endl;
}

TConvolutionalNeuralNetworkCUDA* TConvolutionalNeuralNetworkCUDA::ImportFromONNX(const string& Filename) {
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
    TConvolutionalNeuralNetworkCUDA* cnn = new TConvolutionalNeuralNetworkCUDA(
        inputW, inputH, inputC,
        convFilters, kernelSizes, poolSizes,
        fcSizes, outputSize,
        atReLU, atLinear, ltCrossEntropy,
        0.001, 5.0);
    cnn->UseBatchNorm = (useBN == 1);
    
    // Read conv layer weights
    for (int i = 0; i < numConvLayers; i++) {
        int numFilters;
        file.read(reinterpret_cast<char*>(&numFilters), sizeof(int));
        
        ConvLayerGPU& conv = cnn->ConvLayers[i];
        int weightSize = conv.NumFilters * conv.InputChannels * conv.KernelSize * conv.KernelSize;
        double* h_weights = new double[weightSize];
        double* h_biases = new double[conv.NumFilters];
        
        for (int f = 0; f < numFilters; f++) {
            int d0, d1, d2, d3;
            file.read(reinterpret_cast<char*>(&d0), sizeof(int));
            file.read(reinterpret_cast<char*>(&d1), sizeof(int));
            file.read(reinterpret_cast<char*>(&d2), sizeof(int));
            file.read(reinterpret_cast<char*>(&d3), sizeof(int));
            
            // Read weights for this filter
            int filterOffset = f * conv.InputChannels * conv.KernelSize * conv.KernelSize;
            for (int c = 0; c < conv.InputChannels; c++) {
                for (int ky = 0; ky < conv.KernelSize; ky++) {
                    for (int kx = 0; kx < conv.KernelSize; kx++) {
                        int idx = filterOffset + c * conv.KernelSize * conv.KernelSize + ky * conv.KernelSize + kx;
                        file.read(reinterpret_cast<char*>(&h_weights[idx]), sizeof(double));
                    }
                }
            }
            
            // Read bias
            file.read(reinterpret_cast<char*>(&h_biases[f]), sizeof(double));
        }
        
        CUDA_CHECK(cudaMemcpy(conv.d_Weights, h_weights, weightSize * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(conv.d_Biases, h_biases, conv.NumFilters * sizeof(double), cudaMemcpyHostToDevice));
        
        delete[] h_weights;
        delete[] h_biases;
    }
    
    // Read FC layer weights
    for (int i = 0; i < (int)cnn->FCLayers.size(); i++) {
        int rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        
        FCLayerGPU& fc = cnn->FCLayers[i];
        int weightSize = fc.NumNeurons * fc.NumInputs;
        double* h_weights = new double[weightSize];
        
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                file.read(reinterpret_cast<char*>(&h_weights[r * cols + c]), sizeof(double));
            }
        }
        
        int biasSize;
        file.read(reinterpret_cast<char*>(&biasSize), sizeof(int));
        double* h_biases = new double[biasSize];
        for (int b = 0; b < biasSize; b++) {
            file.read(reinterpret_cast<char*>(&h_biases[b]), sizeof(double));
        }
        
        CUDA_CHECK(cudaMemcpy(fc.d_Weights, h_weights, weightSize * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(fc.d_Biases, h_biases, fc.NumNeurons * sizeof(double), cudaMemcpyHostToDevice));
        
        delete[] h_weights;
        delete[] h_biases;
    }
    
    // Read output layer
    {
        int outRows, outCols;
        file.read(reinterpret_cast<char*>(&outRows), sizeof(int));
        file.read(reinterpret_cast<char*>(&outCols), sizeof(int));
        
        FCLayerGPU& out = cnn->OutputLayer;
        int weightSize = out.NumNeurons * out.NumInputs;
        double* h_weights = new double[weightSize];
        
        for (int r = 0; r < outRows; r++) {
            for (int c = 0; c < outCols; c++) {
                file.read(reinterpret_cast<char*>(&h_weights[r * outCols + c]), sizeof(double));
            }
        }
        
        int outBiasSize;
        file.read(reinterpret_cast<char*>(&outBiasSize), sizeof(int));
        double* h_biases = new double[outBiasSize];
        for (int b = 0; b < outBiasSize; b++) {
            file.read(reinterpret_cast<char*>(&h_biases[b]), sizeof(double));
        }
        
        CUDA_CHECK(cudaMemcpy(out.d_Weights, h_weights, weightSize * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(out.d_Biases, h_biases, out.NumNeurons * sizeof(double), cudaMemcpyHostToDevice));
        
        delete[] h_weights;
        delete[] h_biases;
    }
    
    // Read batch norm params if enabled
    if (cnn->UseBatchNorm) {
        cnn->InitializeBatchNorm();
        for (int i = 0; i < numConvLayers; i++) {
            int size;
            file.read(reinterpret_cast<char*>(&size), sizeof(int));
            
            cnn->FBatchNormParams[i].Initialize(size);
            for (int j = 0; j < size; j++) {
                file.read(reinterpret_cast<char*>(&cnn->FBatchNormParams[i].Gamma[j]), sizeof(double));
                file.read(reinterpret_cast<char*>(&cnn->FBatchNormParams[i].Beta[j]), sizeof(double));
                file.read(reinterpret_cast<char*>(&cnn->FBatchNormParams[i].RunningMean[j]), sizeof(double));
                file.read(reinterpret_cast<char*>(&cnn->FBatchNormParams[i].RunningVar[j]), sizeof(double));
            }
        }
    }
    
    file.close();
    return cnn;
}

// ========== Helper Functions ==========
string ActivationToStr(ActivationType act) {
    switch (act) {
        case atSigmoid: return "sigmoid";
        case atTanh: return "tanh";
        case atReLU: return "relu";
        case atLinear: return "linear";
        default: return "sigmoid";
    }
}

string LossToStr(LossType loss) {
    switch (loss) {
        case ltMSE: return "mse";
        case ltCrossEntropy: return "crossentropy";
        default: return "mse";
    }
}

ActivationType ParseActivation(const string& s) {
    string lower = s;
    transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "tanh") return atTanh;
    else if (lower == "relu") return atReLU;
    else if (lower == "linear") return atLinear;
    else return atSigmoid;
}

LossType ParseLoss(const string& s) {
    string lower = s;
    transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "crossentropy") return ltCrossEntropy;
    else return ltMSE;
}

Command ParseCommand(const string& cmd) {
    if (cmd == "help") return cmdHelp;
    if (cmd == "info") return cmdInfo;
    if (cmd == "create") return cmdCreate;
    if (cmd == "train") return cmdTrain;
    if (cmd == "predict") return cmdPredict;
    if (cmd == "export-onnx") return cmdExportONNX;
    if (cmd == "import-onnx") return cmdImportONNX;
    return cmdNone;
}

vector<int> ParseIntList(const string& str) {
    vector<int> result;
    stringstream iss(str);
    string token;
    while (getline(iss, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t\r\n"));
        token.erase(token.find_last_not_of(" \t\r\n") + 1);
        if (!token.empty()) {
            result.push_back(stoi(token));
        }
    }
    return result;
}

string GetArgValue(int argc, char* argv[], const string& arg, const string& defaultValue = "") {
    string searchKey = arg + "=";
    
    for (int i = 1; i < argc; i++) {
        string argv_str(argv[i]);
        
        if (argv_str.substr(0, searchKey.length()) == searchKey) {
            return argv_str.substr(searchKey.length());
        }
        
        if (argv_str == arg && i + 1 < argc) {
            return string(argv[i + 1]);
        }
    }
    return defaultValue;
}

bool HasArg(int argc, char* argv[], const string& arg) {
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == arg) {
            return true;
        }
    }
    return false;
}

// ========== CLI Helper Functions ==========
void PrintHelp() {
    cout << "Commands:\n";
    cout << "  create      Create a new CNN model and save to JSON\n";
    cout << "  train       Train an existing model with data from JSON\n";
    cout << "  predict     Make predictions with a trained model from JSON\n";
    cout << "  info        Display model information from JSON\n";
    cout << "  export-onnx Export model to ONNX format\n";
    cout << "  import-onnx Import model from ONNX format\n";
    cout << "  help        Show this help message\n\n";
    cout << "Create Options:\n";
    cout << "  --input-w=N            Input width (required)\n";
    cout << "  --input-h=N            Input height (required)\n";
    cout << "  --input-c=N            Input channels (required)\n";
    cout << "  --conv=N,N,...         Conv filters (required)\n";
    cout << "  --kernels=N,N,...      Kernel sizes (required)\n";
    cout << "  --pools=N,N,...        Pool sizes (required)\n";
    cout << "  --fc=N,N,...           FC layer sizes (required)\n";
    cout << "  --output=N             Output layer size (required)\n";
    cout << "  --save=FILE.json       Save model to JSON file (required)\n";
    cout << "  --lr=VALUE             Learning rate (default: 0.001)\n";
    cout << "  --hidden-act=TYPE      sigmoid|tanh|relu|linear (default: relu)\n";
    cout << "  --output-act=TYPE      sigmoid|tanh|relu|linear (default: linear)\n";
    cout << "  --loss=TYPE            mse|crossentropy (default: mse)\n";
    cout << "  --clip=VALUE           Gradient clipping (default: 5.0)\n";
    cout << "  --batch-norm           Enable batch normalization\n\n";
    cout << "Train Options:\n";
    cout << "  --model=FILE.json      Load model from JSON file (required)\n";
    cout << "  --data=FILE.csv        Training data CSV file (required)\n";
    cout << "  --epochs=N             Number of epochs (required)\n";
    cout << "  --save=FILE.json       Save trained model to JSON (required)\n";
    cout << "  --batch-size=N         Batch size (default: 32)\n\n";
    cout << "Predict Options:\n";
    cout << "  --model=FILE.json      Load model from JSON file (required)\n";
    cout << "  --data=FILE.csv        Input data CSV file (required)\n";
    cout << "  --output=FILE.csv      Save predictions to CSV file (required)\n\n";
    cout << "Info Options:\n";
    cout << "  --model=FILE.json      Load model from JSON file (required)\n\n";
    cout << "Export ONNX Options:\n";
    cout << "  --model=FILE.json      Load model from JSON file (required)\n";
    cout << "  --onnx=FILE.onnx       Save to ONNX file (required)\n\n";
    cout << "Import ONNX Options:\n";
    cout << "  --onnx=FILE.onnx       Load from ONNX file (required)\n";
    cout << "  --save=FILE.json       Save to JSON file (required)\n\n";
    cout << "Examples:\n";
    cout << "  cnn create --input-w=28 --input-h=28 --input-c=1 --conv=32,64 --kernels=3,3 --pools=2,2 --fc=128 --output=10 --save=model.json\n";
    cout << "  cnn create --input-w=28 --input-h=28 --input-c=1 --conv=32,64 --kernels=3,3 --pools=2,2 --fc=128 --output=10 --batch-norm --save=model.json\n";
    cout << "  cnn train --model=model.json --data=data.csv --epochs=50 --save=model_trained.json\n";
    cout << "  cnn predict --model=model_trained.json --data=test.csv --output=predictions.csv\n";
    cout << "  cnn info --model=model.json\n";
    cout << "  cnn export-onnx --model=model.json --onnx=model.onnx\n";
    cout << "  cnn import-onnx --onnx=model.onnx --save=imported.json\n";
}

void PrintModelInfo(const string& modelFile) {
    ifstream file(modelFile);
    if (!file.is_open()) {
        cerr << "Error: Cannot open model file: " << modelFile << endl;
        return;
    }

    string content((istreambuf_iterator<char>(file)),
                   istreambuf_iterator<char>());
    file.close();

    auto findValue = [&content](const string& key) -> string {
        string searchKey = "\"" + key + "\": ";
        size_t pos = content.find(searchKey);
        if (pos == string::npos) return "";
        pos += searchKey.length();
        while (pos < content.length() && (content[pos] == ' ' || content[pos] == '\n')) pos++;
        size_t endPos = pos;
        while (endPos < content.length() && content[endPos] != ',' &&
            content[endPos] != '\n' && content[endPos] != '}') endPos++;
        return content.substr(pos, endPos - pos);
    };

    cout << "\n=================================================================\n";
    cout << "  Model Information:  " << modelFile << "\n";
    cout << "=================================================================\n\n";
    cout << "Architecture:\n";
    cout << "Input: " << findValue("input_width") << "x"
         << findValue("input_height") << "x"
         << findValue("input_channels") << "\n";
    cout << "Output size: " << findValue("output_size") << "\n\n";
    cout << "Training Parameters:\n";
    cout << "Learning rate: " << findValue("learning_rate") << "\n";
    cout << "Gradient clip: " << findValue("gradient_clip") << "\n";
    cout << "activation: " << findValue("activation") << "\n";
    cout << "output_activation: " << findValue("output_activation") << "\n";
    cout << "loss_type: " << findValue("loss_type") << "\n\n";
}

void HandleCreate(int argc, char* argv[]) {
    string saveFile = GetArgValue(argc, argv, "--save", "");
    string inputWStr = GetArgValue(argc, argv, "--input-w", "");
    string inputHStr = GetArgValue(argc, argv, "--input-h", "");
    string inputCStr = GetArgValue(argc, argv, "--input-c", "");
    string convFilters = GetArgValue(argc, argv, "--conv", "");
    string kernels = GetArgValue(argc, argv, "--kernels", "");
    string pools = GetArgValue(argc, argv, "--pools", "");
    string fcLayers = GetArgValue(argc, argv, "--fc", "");
    string outputSizeStr = GetArgValue(argc, argv, "--output", "");

    if (saveFile.empty()) {
        cerr << "Error: --save argument is required for create command" << endl;
        return;
    }
    if (inputWStr.empty()) {
        cerr << "Error: --input-w argument is required for create command" << endl;
        return;
    }
    if (inputHStr.empty()) {
        cerr << "Error: --input-h argument is required for create command" << endl;
        return;
    }
    if (inputCStr.empty()) {
        cerr << "Error: --input-c argument is required for create command" << endl;
        return;
    }
    if (convFilters.empty()) {
        cerr << "Error: --conv argument is required for create command" << endl;
        return;
    }
    if (kernels.empty()) {
        cerr << "Error: --kernels argument is required for create command" << endl;
        return;
    }
    if (pools.empty()) {
        cerr << "Error: --pools argument is required for create command" << endl;
        return;
    }
    if (fcLayers.empty()) {
        cerr << "Error: --fc argument is required for create command" << endl;
        return;
    }
    if (outputSizeStr.empty()) {
        cerr << "Error: --output argument is required for create command" << endl;
        return;
    }

    int inputW = stoi(inputWStr);
    int inputH = stoi(inputHStr);
    int inputC = stoi(inputCStr);
    int outputSize = stoi(outputSizeStr);

    string hiddenActStr = GetArgValue(argc, argv, "--hidden-act", "relu");
    string outputActStr = GetArgValue(argc, argv, "--output-act", "linear");
    string lossStr = GetArgValue(argc, argv, "--loss", "mse");
    double lr = stod(GetArgValue(argc, argv, "--lr", "0.001"));
    double clip = stod(GetArgValue(argc, argv, "--clip", "5.0"));
    bool useBatchNorm = HasArg(argc, argv, "--batch-norm");

    vector<int> convFilterVec = ParseIntList(convFilters);
    vector<int> kernelVec = ParseIntList(kernels);
    vector<int> poolVec = ParseIntList(pools);
    vector<int> fcVec = ParseIntList(fcLayers);

    ActivationType hiddenAct = ParseActivation(hiddenActStr);
    ActivationType outputAct = ParseActivation(outputActStr);
    LossType lossType = ParseLoss(lossStr);

    cout << "Creating CNN model...\n";
    cout << "  Input: " << inputW << "x" << inputH << "x" << inputC << "\n";
    
    cout << "  Conv filters: ";
    for (size_t i = 0; i < convFilterVec.size(); i++) {
        if (i > 0) cout << ",";
        cout << convFilterVec[i];
    }
    cout << "\n";
    
    cout << "  Kernel sizes: ";
    for (size_t i = 0; i < kernelVec.size(); i++) {
        if (i > 0) cout << ",";
        cout << kernelVec[i];
    }
    cout << "\n";
    
    cout << "  Pool sizes: ";
    for (size_t i = 0; i < poolVec.size(); i++) {
        if (i > 0) cout << ",";
        cout << poolVec[i];
    }
    cout << "\n";
    
    cout << "  FC layers: ";
    for (size_t i = 0; i < fcVec.size(); i++) {
        if (i > 0) cout << ",";
        cout << fcVec[i];
    }
    cout << "\n";
    
    cout << "Output size: " << outputSize << "\n";
    cout << "  Hidden activation: " << ActivationToStr(hiddenAct) << "\n";
    cout << "  Output activation: " << ActivationToStr(outputAct) << "\n";
    cout << "  Loss function: " << LossToStr(lossType) << "\n";
    cout << fixed << setprecision(6) << "  Learning rate: " << lr << "\n";
    cout << fixed << setprecision(2) << "  Gradient clip: " << clip << "\n";
    cout << "  Batch normalization: " << (useBatchNorm ? "enabled" : "disabled") << "\n";

    TAdvancedCNN* cnn = new TAdvancedCNN(
        inputW, inputH, inputC,
        convFilterVec, kernelVec, poolVec, fcVec,
        outputSize, hiddenAct, outputAct,
        lossType, lr, clip
    );
    
    if (useBatchNorm) {
        cnn->UseBatchNorm = true;
        cnn->InitializeBatchNorm();
    }

    if (cnn->SaveToJSON(saveFile.c_str())) {
        cout << "Created CNN model\n";
        cout << "Model saved to: " << saveFile << "\n";
    } else {
        cerr << "Error: Failed to save model to " << saveFile << "\n";
    }

    delete cnn;
}

void HandleTrain(int argc, char* argv[]) {
    string modelFile = GetArgValue(argc, argv, "--model", "");
    string dataFile = GetArgValue(argc, argv, "--data", "");
    string epochsStr = GetArgValue(argc, argv, "--epochs", "");
    string saveFile = GetArgValue(argc, argv, "--save", "");
    int batchSize = stoi(GetArgValue(argc, argv, "--batch-size", "32"));

    if (modelFile.empty()) {
        cerr << "Error: --model argument is required for train command" << endl;
        return;
    }
    if (dataFile.empty()) {
        cerr << "Error: --data argument is required for train command" << endl;
        return;
    }
    if (epochsStr.empty()) {
        cerr << "Error: --epochs argument is required for train command" << endl;
        return;
    }
    if (saveFile.empty()) {
        cerr << "Error: --save argument is required for train command" << endl;
        return;
    }

    int epochs = stoi(epochsStr);

    cout << "Training model...\n";
    cout << "  Model: " << modelFile << "\n";
    cout << "  Data: " << dataFile << "\n";
    cout << "  Epochs: " << epochs << "\n";
    cout << "  Batch size: " << batchSize << "\n";
    cout << "  Save to: " << saveFile << "\n\n";

    cout << "Training not fully implemented in this CLI demo.\n";
    cout << "To implement training:\n";
    cout << "  1. Load CSV data from " << dataFile << "\n";
    cout << "  2. Load model from " << modelFile << "\n";
    cout << "  3. Run training loop with TrainStep() for " << epochs << " epochs\n";
    cout << "  4. Save updated model to " << saveFile << "\n";
    cout << "\nSee the library API for complete training implementation.\n";
}

void HandlePredict(int argc, char* argv[]) {
    string modelFile = GetArgValue(argc, argv, "--model", "");
    string dataFile = GetArgValue(argc, argv, "--data", "");
    string outputFile = GetArgValue(argc, argv, "--output", "");

    if (modelFile.empty()) {
        cerr << "Error: --model argument is required for predict command" << endl;
        return;
    }
    if (dataFile.empty()) {
        cerr << "Error: --data argument is required for predict command" << endl;
        return;
    }
    if (outputFile.empty()) {
        cerr << "Error: --output argument is required for predict command" << endl;
        return;
    }

    cout << "Making predictions...\n";
    cout << "  Model: " << modelFile << "\n";
    cout << "  Data: " << dataFile << "\n";
    cout << "  Output: " << outputFile << "\n\n";

    cout << "Prediction not fully implemented in this CLI demo.\n";
    cout << "To implement prediction:\n";
    cout << "  1. Load model from " << modelFile << "\n";
    cout << "  2. Load input data from CSV file: " << dataFile << "\n";
    cout << "  3. Run Predict() on each input\n";
    cout << "  4. Save predictions to CSV: " << outputFile << "\n";
    cout << "\nSee the library API for complete prediction implementation.\n";
}

// ========== Main Program ==========
int main(int argc, char* argv[]) {
    srand(static_cast<unsigned int>(time(nullptr)));

    if (argc < 2) {
        PrintHelp();
        return 0;
    }

    if (string(argv[1]) == "--help" || string(argv[1]) == "-h") {
        PrintHelp();
        return 0;
    }

    Command cmd = ParseCommand(argv[1]);

    try {
        switch (cmd) {
            case cmdHelp:
                PrintHelp();
                break;

            case cmdInfo: {
                string modelFile = GetArgValue(argc, argv, "--model", "");
                if (modelFile.empty()) {
                    cerr << "Error: --model argument required for info command\n";
                    return 1;
                }
                PrintModelInfo(modelFile);
                break;
            }

            case cmdCreate:
                HandleCreate(argc, argv);
                break;

            case cmdTrain:
                HandleTrain(argc, argv);
                break;

            case cmdPredict:
                HandlePredict(argc, argv);
                break;

            case cmdExportONNX: {
                string modelFile = GetArgValue(argc, argv, "--model", "");
                string onnxFile = GetArgValue(argc, argv, "--onnx", "");
                if (modelFile.empty()) {
                    cerr << "Error: --model argument required for export-onnx command\n";
                    return 1;
                }
                if (onnxFile.empty()) {
                    cerr << "Error: --onnx argument required for export-onnx command\n";
                    return 1;
                }
                
                TAdvancedCNN* cnn = TAdvancedCNN::Load(modelFile.c_str());
                if (!cnn) {
                    cerr << "Error: Could not load model from " << modelFile << "\n";
                    return 1;
                }
                cnn->ExportToONNX(onnxFile);
                delete cnn;
                break;
            }

            case cmdImportONNX: {
                string onnxFile = GetArgValue(argc, argv, "--onnx", "");
                string saveFile = GetArgValue(argc, argv, "--save", "");
                if (onnxFile.empty()) {
                    cerr << "Error: --onnx argument required for import-onnx command\n";
                    return 1;
                }
                if (saveFile.empty()) {
                    cerr << "Error: --save argument required for import-onnx command\n";
                    return 1;
                }
                
                TAdvancedCNN* cnn = TAdvancedCNN::ImportFromONNX(onnxFile);
                cnn->SaveToJSON(saveFile.c_str());
                cout << "Model imported from ONNX and saved to: " << saveFile << endl;
                delete cnn;
                break;
            }

            case cmdNone:
            default:
                cerr << "Unknown command: '" << argv[1] << "'\n";
                cerr << "Run 'cnn help' for usage information\n";
                return 1;
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
