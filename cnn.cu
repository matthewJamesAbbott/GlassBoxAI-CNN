//
// Matthew Abbott 2025
// Advanced CNN with Full Backpropagation, Adam Optimizer, Batch Processing
// CUDA Implementation
//

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
    cmdHelp
};

// Type aliases matching Pascal
typedef vector<double> DArray;
typedef vector<DArray> TDArray2D;
typedef vector<TDArray2D> TDArray3D;
typedef vector<TDArray3D> TDArray4D;
typedef vector<int> TIntArray;

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
    double Beta1, Beta2;
    int AdamT;
    bool IsTraining;

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
                                    double learningRate = 0.001,
                                    double dropoutRate = 0.25);
    ~TConvolutionalNeuralNetworkCUDA();

    void Predict(const double* imageData, double* result);
    double TrainStep(const double* imageData, const double* target);
    bool Save(const char* filename);
    static TConvolutionalNeuralNetworkCUDA* Load(const char* filename);

    int GetInputWidth() const { return InputWidth; }
    int GetInputHeight() const { return InputHeight; }
    int GetInputChannels() const { return InputChannels; }
    int GetOutputSize() const { return OutputSize; }
    int GetNumConvLayers() const { return ConvLayers.size(); }
    int GetNumFCLayers() const { return FCLayers.size(); }
    double GetLearningRate() const { return LearningRate; }
    double GetDropoutRate() const { return DropoutRate; }
    const std::vector<int>& GetConvFilters() const { return FConvFilters; }
    const std::vector<int>& GetKernelSizes() const { return FKernelSizes; }
    const std::vector<int>& GetPoolSizes() const { return FPoolSizes; }
    const std::vector<int>& GetFCSizes() const { return FFCSizes; }
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
    double learningRate,
    double dropoutRate)
{
    LearningRate = learningRate;
    DropoutRate = dropoutRate;
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
        learningRate, dropoutRate);

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

// Helper functions
void ParseIntArrayHelper(const string& s, TIntArray& result) {
    result.clear();
    stringstream ss(s);
    string token;
    while (getline(ss, token, ',')) {
        // Trim whitespace
        token.erase(0, token.find_first_not_of(" \t\r\n"));
        token.erase(token.find_last_not_of(" \t\r\n") + 1);
        if (!token.empty()) {
            result.push_back(stoi(token));
        }
    }
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

void PrintUsage() {
    cout << "CNN - Command-line Convolutional Neural Network\n";
    cout << "Matthew Abbott 2025 \n\n";
    cout << "Commands:\n";
    cout << "  create   Create a new CNN model\n";
    cout << "  train    Train an existing model with data\n";
    cout << "  predict  Make predictions with a trained model\n";
    cout << "  info     Display model information\n";
    cout << "  help     Show this help message\n\n";
    cout << "Create Options:\n";
    cout << "  --input-w=N            Input width (required)\n";
    cout << "  --input-h=N            Input height (required)\n";
    cout << "  --input-c=N            Input channels (required)\n";
    cout << "  --conv=N,N,...         Conv filters (required)\n";
    cout << "  --kernels=N,N,...      Kernel sizes (required)\n";
    cout << "  --pools=N,N,...        Pool sizes (required)\n";
    cout << "  --fc=N,N,...           FC layer sizes (required)\n";
    cout << "  --output=N             Output layer size (required)\n";
    cout << "  --save=FILE            Save model to file (required)\n";
    cout << "  --lr=VALUE             Learning rate (default: 0.001)\n";
    cout << "  --hidden-act=TYPE      sigmoid|tanh|relu|linear (default: relu)\n";
    cout << "  --output-act=TYPE      sigmoid|tanh|relu|linear (default: linear)\n";
    cout << "  --loss=TYPE            mse|crossentropy (default: mse)\n";
    cout << "  --clip=VALUE           Gradient clipping (default: 5.0)\n\n";
    cout << "Examples:\n";
    cout << "  cnn create --input-w=28 --input-h=28 --input-c=1 --conv=32,64 --kernels=3,3 --pools=2,2 --fc=128 --output=10 --save=model.bin\n";
    cout << "  cnn train --model=model.bin --data=data.csv --epochs=50 --save=model_trained.bin\n";
}

int main(int argc, char* argv[]) {
    srand(time(0));

    if (argc < 2) {
        PrintUsage();
        return 0;
    }

    string CmdStr = argv[1];
    Command Cmd = cmdNone;

    if (CmdStr == "create") Cmd = cmdCreate;
    else if (CmdStr == "train") Cmd = cmdTrain;
    else if (CmdStr == "predict") Cmd = cmdPredict;
    else if (CmdStr == "info") Cmd = cmdInfo;
    else if (CmdStr == "help" || CmdStr == "--help" || CmdStr == "-h") Cmd = cmdHelp;
    else {
        cerr << "Unknown command: " << CmdStr << "\n";
        PrintUsage();
        return 1;
    }

    if (Cmd == cmdHelp) {
        PrintUsage();
        return 0;
    }

    // Initialize defaults
    int inputW = 0, inputH = 0, inputC = 0, outputSize = 0;
    TIntArray convFilters, kernelSizes, poolSizes, fcLayerSizes;
    double learningRate = 0.001;
    double gradientClip = 5.0;
    ActivationType hiddenAct = atReLU;
    ActivationType outputAct = atLinear;
    LossType lossType = ltMSE;
    string modelFile = "";
    string saveFile = "";

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        string arg = argv[i];
        size_t eqPos = arg.find('=');
        if (eqPos == string::npos) {
            cerr << "Invalid argument: " << arg << "\n";
            continue;
        }

        string key = arg.substr(0, eqPos);
        string value = arg.substr(eqPos + 1);

        if (key == "--input-w") inputW = stoi(value);
        else if (key == "--input-h") inputH = stoi(value);
        else if (key == "--input-c") inputC = stoi(value);
        else if (key == "--output") outputSize = stoi(value);
        else if (key == "--conv") ParseIntArrayHelper(value, convFilters);
        else if (key == "--kernels") ParseIntArrayHelper(value, kernelSizes);
        else if (key == "--pools") ParseIntArrayHelper(value, poolSizes);
        else if (key == "--fc") ParseIntArrayHelper(value, fcLayerSizes);
        else if (key == "--save") saveFile = value;
        else if (key == "--model") modelFile = value;
        else if (key == "--lr") learningRate = stod(value);
        else if (key == "--hidden-act") hiddenAct = ParseActivation(value);
        else if (key == "--output-act") outputAct = ParseActivation(value);
        else if (key == "--loss") lossType = ParseLoss(value);
        else if (key == "--clip") gradientClip = stod(value);
        else cerr << "Unknown option: " << key << "\n";
    }

    // Execute command
    if (Cmd == cmdCreate) {
        if (inputW <= 0) { cerr << "Error: --input-w is required\n"; return 1; }
        if (inputH <= 0) { cerr << "Error: --input-h is required\n"; return 1; }
        if (inputC <= 0) { cerr << "Error: --input-c is required\n"; return 1; }
        if (convFilters.empty()) { cerr << "Error: --conv is required\n"; return 1; }
        if (kernelSizes.empty()) { cerr << "Error: --kernels is required\n"; return 1; }
        if (poolSizes.empty()) { cerr << "Error: --pools is required\n"; return 1; }
        if (fcLayerSizes.empty()) { cerr << "Error: --fc is required\n"; return 1; }
        if (outputSize <= 0) { cerr << "Error: --output is required\n"; return 1; }
        if (saveFile.empty()) { cerr << "Error: --save is required\n"; return 1; }

        TAdvancedCNN* CNN = new TAdvancedCNN(inputW, inputH, inputC, convFilters, kernelSizes,
                                             poolSizes, fcLayerSizes, outputSize,
                                             hiddenAct, outputAct, lossType, 
                                             learningRate, gradientClip);

        cout << "Created CNN model:\n";
        cout << "  Input: " << inputW << "x" << inputH << "x" << inputC << "\n";
        
        cout << "  Conv filters: ";
        for (size_t i = 0; i < convFilters.size(); i++) {
            if (i > 0) cout << ",";
            cout << convFilters[i];
        }
        cout << "\n";
        
        cout << "  Kernel sizes: ";
        for (size_t i = 0; i < kernelSizes.size(); i++) {
            if (i > 0) cout << ",";
            cout << kernelSizes[i];
        }
        cout << "\n";
        
        cout << "  Pool sizes: ";
        for (size_t i = 0; i < poolSizes.size(); i++) {
            if (i > 0) cout << ",";
            cout << poolSizes[i];
        }
        cout << "\n";
        
        cout << "  FC layers: ";
        for (size_t i = 0; i < fcLayerSizes.size(); i++) {
            if (i > 0) cout << ",";
            cout << fcLayerSizes[i];
        }
        cout << "\n";
        
        cout << "  Output size: " << outputSize << "\n";
        cout << "  Hidden activation: " << ActivationToStr(hiddenAct) << "\n";
        cout << "  Output activation: " << ActivationToStr(outputAct) << "\n";
        cout << "  Loss function: " << LossToStr(lossType) << "\n";
        cout << "  Learning rate: " << fixed << setprecision(6) << learningRate << "\n";
        cout << "  Gradient clip: " << fixed << setprecision(2) << gradientClip << "\n";
        cout << "  Saved to: " << saveFile << "\n";

        delete CNN;
    }
    else if (Cmd == cmdTrain) {
        cout << "Train command requires model persistence (not yet implemented)\n";
    }
    else if (Cmd == cmdPredict) {
        cout << "Predict command requires model persistence (not yet implemented)\n";
    }
    else if (Cmd == cmdInfo) {
        cout << "Info command requires model persistence (not yet implemented)\n";
    }

    return 0;
}
