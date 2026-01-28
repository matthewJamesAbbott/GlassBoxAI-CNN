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

#ifndef CNN_OPENCL_H
#define CNN_OPENCL_H

#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <iomanip>
#include <iostream>
#include <ctime>
#include <CL/cl.h>

// Type definitions to match Pascal
using DArray = std::vector<double>;
using TDArray2D = std::vector<DArray>;
using TDArray3D = std::vector<TDArray2D>;
using TDArray4D = std::vector<TDArray3D>;
using TIntArray = std::vector<int>;

// Enumerations
enum class TActivationType {
    atSigmoid,
    atTanh,
    atReLU,
    atLinear
};

enum class TLossType {
    ltMSE,
    ltCrossEntropy
};

enum class TPaddingType {
    ptSame,
    ptValid
};

enum class TCommand {
    cmdNone,
    cmdCreate,
    cmdTrain,
    cmdPredict,
    cmdInfo,
    cmdHelp,
    cmdExportONNX,
    cmdImportONNX
};

// Batch Normalization Parameters
struct TBatchNormParams {
    DArray Gamma;
    DArray Beta;
    DArray RunningMean;
    DArray RunningVar;
    double Epsilon = 1e-5;
    double Momentum = 0.1;
    
    TBatchNormParams() : Epsilon(1e-5), Momentum(0.1) {}
    
    void Initialize(int Size) {
        Gamma.resize(Size, 1.0);
        Beta.resize(Size, 0.0);
        RunningMean.resize(Size, 0.0);
        RunningVar.resize(Size, 1.0);
    }
};

// Data split structure
struct TDataSplit {
    TDArray4D TrainInputs;
    TDArray3D TrainTargets;
    TDArray4D ValInputs;
    TDArray3D ValTargets;
};

// Forward declarations
class TConvFilter;
class TConvLayer;
class TPoolingLayer;
class TFCLayer;
class TAdvancedCNN;

// ========== OpenCL Utilities ==========
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

// ========== Utility Functions ==========
inline double ClipValue(double V, double MaxVal) {
    if (V > MaxVal) return MaxVal;
    else if (V < -MaxVal) return -MaxVal;
    else return V;
}

inline double RandomWeight(double Scale) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(-1.0, 1.0);
    return dis(gen) * Scale;
}

inline void InitMatrix(TDArray2D& M, int Rows, int Cols, double Scale) {
    M.resize(Rows);
    for (int i = 0; i < Rows; i++) {
        M[i].resize(Cols);
        for (int j = 0; j < Cols; j++) {
            M[i][j] = RandomWeight(Scale);
        }
    }
}

inline void ZeroMatrix(TDArray2D& M, int Rows, int Cols) {
    M.resize(Rows);
    for (int i = 0; i < Rows; i++) {
        M[i].resize(Cols);
        for (int j = 0; j < Cols; j++) {
            M[i][j] = 0.0;
        }
    }
}

inline void ZeroArray(DArray& A, int Size) {
    A.resize(Size);
    for (int i = 0; i < Size; i++) {
        A[i] = 0.0;
    }
}

inline void Zero3DArray(TDArray3D& A, int D1, int D2, int D3) {
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

inline void Zero4DArray(TDArray4D& A, int D1, int D2, int D3, int D4) {
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

// ========== TActivation Class ==========
class TActivation {
public:
    static double Apply(double X, TActivationType ActType) {
        switch (ActType) {
            case TActivationType::atSigmoid:
                return 1.0 / (1.0 + std::exp(-std::max(-500.0, std::min(500.0, X))));
            case TActivationType::atTanh:
                return std::tanh(X);
            case TActivationType::atReLU:
                return (X > 0) ? X : 0;
            case TActivationType::atLinear:
                return X;
            default:
                return X;
        }
    }

    static double Derivative(double Y, TActivationType ActType) {
        switch (ActType) {
            case TActivationType::atSigmoid:
                return Y * (1.0 - Y);
            case TActivationType::atTanh:
                return 1.0 - Y * Y;
            case TActivationType::atReLU:
                return (Y > 0) ? 1.0 : 0.0;
            case TActivationType::atLinear:
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
            Arr[i] = std::exp(Arr[i] - MaxVal);
            Sum += Arr[i];
        }

        for (size_t i = 0; i < Arr.size(); i++) {
            Arr[i] = Arr[i] / Sum;
        }
    }
};

#endif // CNN_OPENCL_H

// ========== TLoss Class ==========
class TLoss {
public:
    static double Compute(const DArray& Pred, const DArray& Target, TLossType LossType) {
        double Result = 0;

        switch (LossType) {
            case TLossType::ltMSE:
                for (size_t i = 0; i < Pred.size(); i++) {
                    Result += (Pred[i] - Target[i]) * (Pred[i] - Target[i]);
                }
                break;

            case TLossType::ltCrossEntropy:
                for (size_t i = 0; i < Pred.size(); i++) {
                    double P = std::max(1e-15, std::min(1.0 - 1e-15, Pred[i]));
                    Result += -(Target[i] * std::log(P) + (1.0 - Target[i]) * std::log(1.0 - P));
                }
                break;
        }

        return Result / Pred.size();
    }

    static void Gradient(const DArray& Pred, const DArray& Target, TLossType LossType, DArray& Grad) {
        Grad.resize(Pred.size());

        switch (LossType) {
            case TLossType::ltMSE:
                for (size_t i = 0; i < Pred.size(); i++) {
                    Grad[i] = Pred[i] - Target[i];
                }
                break;

            case TLossType::ltCrossEntropy:
                for (size_t i = 0; i < Pred.size(); i++) {
                    double P = std::max(1e-15, std::min(1.0 - 1e-15, Pred[i]));
                    Grad[i] = (P - Target[i]) / (P * (1.0 - P) + 1e-15);
                }
                break;
        }
    }
};

// ========== TConvFilter Class ==========
class TConvFilter {
private:
    int FInputChannels;
    int FOutputChannels;
    int FKernelSize;

public:
    TDArray4D Weights;
    double Bias;
    TDArray4D dWeights;
    double dBias;
    
    cl_mem WeightsBuffer;
    cl_mem dWeightsBuffer;

    TConvFilter(int InputChannels, int OutputChannels, int KernelSize)
        : FInputChannels(InputChannels),
          FOutputChannels(OutputChannels),
          FKernelSize(KernelSize),
          Bias(0.0),
          dBias(0.0),
          WeightsBuffer(nullptr),
          dWeightsBuffer(nullptr) {

        double Scale = std::sqrt(2.0 / (InputChannels * KernelSize * KernelSize));

        Zero4DArray(Weights, OutputChannels, InputChannels, KernelSize, KernelSize);
        Zero4DArray(dWeights, OutputChannels, InputChannels, KernelSize, KernelSize);

        // Initialize weights with random values
        for (int i = 0; i < OutputChannels; i++) {
            for (int j = 0; j < InputChannels; j++) {
                for (int k = 0; k < KernelSize; k++) {
                    for (int l = 0; l < KernelSize; l++) {
                        Weights[i][j][k][l] = RandomWeight(Scale);
                    }
                }
            }
        }
        
        AllocateGPUBuffers();
    }
    
    ~TConvFilter() {
        if (WeightsBuffer) clReleaseMemObject(WeightsBuffer);
        if (dWeightsBuffer) clReleaseMemObject(dWeightsBuffer);
    }
    
    void AllocateGPUBuffers() {
        cl_int err;
        size_t weightsSize = Weights.size() * Weights[0].size() * 
                            Weights[0][0].size() * Weights[0][0][0].size() * sizeof(double);
        
        WeightsBuffer = clCreateBuffer(gOpenCLContext.context, CL_MEM_READ_WRITE, 
                                      weightsSize, nullptr, &err);
        dWeightsBuffer = clCreateBuffer(gOpenCLContext.context, CL_MEM_READ_WRITE,
                                       weightsSize, nullptr, &err);
    }
    
    void CopyWeightsToGPU() {
        std::vector<double> flatWeights;
        for (size_t i = 0; i < Weights.size(); i++) {
            for (size_t j = 0; j < Weights[i].size(); j++) {
                for (size_t k = 0; k < Weights[i][j].size(); k++) {
                    for (size_t l = 0; l < Weights[i][j][k].size(); l++) {
                        flatWeights.push_back(Weights[i][j][k][l]);
                    }
                }
            }
        }
        clEnqueueWriteBuffer(gOpenCLContext.queue, WeightsBuffer, CL_TRUE, 0,
                           flatWeights.size() * sizeof(double), flatWeights.data(), 0, nullptr, nullptr);
    }
    
    void CopyWeightsFromGPU() {
        std::vector<double> flatWeights(Weights.size() * Weights[0].size() * 
                                       Weights[0][0].size() * Weights[0][0][0].size());
        clEnqueueReadBuffer(gOpenCLContext.queue, WeightsBuffer, CL_TRUE, 0,
                          flatWeights.size() * sizeof(double), flatWeights.data(), 0, nullptr, nullptr);
        
        size_t idx = 0;
        for (size_t i = 0; i < Weights.size(); i++) {
            for (size_t j = 0; j < Weights[i].size(); j++) {
                for (size_t k = 0; k < Weights[i][j].size(); k++) {
                    for (size_t l = 0; l < Weights[i][j][k].size(); l++) {
                        Weights[i][j][k][l] = flatWeights[idx++];
                    }
                }
            }
        }
    }

    void ResetGradients() {
        for (size_t i = 0; i < dWeights.size(); i++) {
            for (size_t j = 0; j < dWeights[i].size(); j++) {
                for (size_t k = 0; k < dWeights[i][j].size(); k++) {
                    for (size_t l = 0; l < dWeights[i][j][k].size(); l++) {
                        dWeights[i][j][k][l] = 0.0;
                    }
                }
            }
        }
        dBias = 0.0;
    }
};

// ========== TConvLayer Class ==========
class TConvLayer {
private:
    int FInputChannels;
    int FOutputChannels;
    int FKernelSize;
    int FStride;
    int FPadding;
    TActivationType FActivation;

    TDArray3D Pad3D(const TDArray3D& Input, int Padding) {
        if (Padding == 0) {
            return Input;
        }

        int Channels = Input.size();
        int Height = Input[0].size();
        int Width = Input[0][0].size();
        int NewHeight = Height + 2 * Padding;
        int NewWidth = Width + 2 * Padding;

        TDArray3D Result;
        Result.resize(Channels);
        for (int c = 0; c < Channels; c++) {
            Result[c].resize(NewHeight);
            for (int h = 0; h < NewHeight; h++) {
                Result[c][h].resize(NewWidth);
                for (int w = 0; w < NewWidth; w++) {
                    int SrcH = h - Padding;
                    int SrcW = w - Padding;
                    if (SrcH >= 0 && SrcH < Height && SrcW >= 0 && SrcW < Width) {
                        Result[c][h][w] = Input[c][SrcH][SrcW];
                    } else {
                        Result[c][h][w] = 0.0;
                    }
                }
            }
        }

        return Result;
    }

public:
    std::vector<TConvFilter*> Filters;
    TDArray3D InputCache;
    TDArray3D OutputCache;
    TDArray3D PreActivation;
    
    cl_mem InputBuffer;
    cl_mem OutputBuffer;
    cl_mem PreActivationBuffer;

    TConvLayer(int InputChannels, int OutputChannels, int KernelSize,
               int Stride, int Padding, TActivationType Activation)
        : FInputChannels(InputChannels),
          FOutputChannels(OutputChannels),
          FKernelSize(KernelSize),
          FStride(Stride),
          FPadding(Padding),
          FActivation(Activation),
          InputBuffer(nullptr),
          OutputBuffer(nullptr),
          PreActivationBuffer(nullptr) {

        Filters.resize(OutputChannels);
        for (int i = 0; i < OutputChannels; i++) {
            Filters[i] = new TConvFilter(InputChannels, 1, KernelSize);
        }
    }

    ~TConvLayer() {
        for (size_t i = 0; i < Filters.size(); i++) {
            delete Filters[i];
        }
        if (InputBuffer) clReleaseMemObject(InputBuffer);
        if (OutputBuffer) clReleaseMemObject(OutputBuffer);
        if (PreActivationBuffer) clReleaseMemObject(PreActivationBuffer);
    }

    void Forward(const TDArray3D& Input, TDArray3D& Output) {
        InputCache = Input;

        TDArray3D Padded;
        if (FPadding > 0) {
            Padded = Pad3D(Input, FPadding);
        } else {
            Padded = Input;
        }

        int outH = (Padded[0].size() - FKernelSize) / FStride + 1;
        int outW = (Padded[0][0].size() - FKernelSize) / FStride + 1;

        Zero3DArray(Output, FOutputChannels, outH, outW);
        Zero3DArray(PreActivation, FOutputChannels, outH, outW);

        // CPU fallback for convolution (can be optimized with custom OpenCL kernels)
        for (int f = 0; f < FOutputChannels; f++) {
            Filters[f]->CopyWeightsToGPU();
            
            for (int h = 0; h < outH; h++) {
                for (int w = 0; w < outW; w++) {
                    double Sum = Filters[f]->Bias;

                    for (int c = 0; c < FInputChannels; c++) {
                        for (int kh = 0; kh < FKernelSize; kh++) {
                            for (int kw = 0; kw < FKernelSize; kw++) {
                                int ih = h * FStride + kh;
                                int iw = w * FStride + kw;
                                Sum += Padded[c][ih][iw] * Filters[f]->Weights[0][c][kh][kw];
                            }
                        }
                    }

                    PreActivation[f][h][w] = Sum;
                    Output[f][h][w] = TActivation::Apply(Sum, FActivation);
                }
            }
        }

        OutputCache = Output;
    }

    void Backward(const TDArray3D& dOutput, TDArray3D& dInput);
    void ApplyGradients(double LR, double ClipVal);
    void ResetGradients();
    int GetOutputChannels();
};

// ========== TConvLayer Class (continued) ==========

void TConvLayer::Backward(const TDArray3D& dOutput, TDArray3D& dInput) {
    int outH = dOutput[0].size();
    int outW = dOutput[0][0].size();

    TDArray3D dH;
    Zero3DArray(dH, FOutputChannels, outH, outW);

    // Compute activation derivatives
    for (int f = 0; f < FOutputChannels; f++) {
        for (int h = 0; h < outH; h++) {
            for (int w = 0; w < outW; w++) {
                dH[f][h][w] = dOutput[f][h][w] *
                              TActivation::Derivative(OutputCache[f][h][w], FActivation);
            }
        }
    }

    // Compute bias gradients
    for (int f = 0; f < FOutputChannels; f++) {
        double Sum = 0.0;
        for (int h = 0; h < outH; h++) {
            for (int w = 0; w < outW; w++) {
                Sum += dH[f][h][w];
            }
        }
        Filters[f]->dBias = Sum;
    }

    // Compute weight gradients
    for (int f = 0; f < FOutputChannels; f++) {
        for (int c = 0; c < FInputChannels; c++) {
            for (int kh = 0; kh < FKernelSize; kh++) {
                for (int kw = 0; kw < FKernelSize; kw++) {
                    double Sum = 0.0;

                    for (int h = 0; h < outH; h++) {
                        for (int w = 0; w < outW; w++) {
                            int ih = h * FStride + kh;
                            int iw = w * FStride + kw;

                            if (FPadding > 0) {
                                Sum += dH[f][h][w] * InputCache[c][ih - FPadding][iw - FPadding];
                            } else {
                                Sum += dH[f][h][w] * InputCache[c][ih][iw];
                            }
                        }
                    }

                    Filters[f]->dWeights[0][c][kh][kw] = Sum;
                }
            }
        }
    }

    // Compute input gradients
    if (InputCache.size() > 0) {
        Zero3DArray(dInput, FInputChannels, InputCache[0].size(), InputCache[0][0].size());

        for (int f = 0; f < FOutputChannels; f++) {
            for (int h = 0; h < outH; h++) {
                for (int w = 0; w < outW; w++) {
                    for (int c = 0; c < FInputChannels; c++) {
                        for (int kh = 0; kh < FKernelSize; kh++) {
                            for (int kw = 0; kw < FKernelSize; kw++) {
                                int ih = h * FStride + kh;
                                int iw = w * FStride + kw;

                                if (ih >= 0 && ih < (int)dInput[c].size() &&
                                    iw >= 0 && iw < (int)dInput[c][0].size()) {
                                    dInput[c][ih][iw] += dH[f][h][w] *
                                                         Filters[f]->Weights[0][c][kh][kw];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void TConvLayer::ApplyGradients(double LR, double ClipVal) {
    for (int f = 0; f < FOutputChannels; f++) {
        Filters[f]->Bias -= LR * ClipValue(Filters[f]->dBias, ClipVal);

        for (size_t i = 0; i < Filters[f]->Weights[0].size(); i++) {
            for (size_t j = 0; j < Filters[f]->Weights[0][i].size(); j++) {
                for (size_t k = 0; k < Filters[f]->Weights[0][i][j].size(); k++) {
                    Filters[f]->Weights[0][i][j][k] -=
                        LR * ClipValue(Filters[f]->dWeights[0][i][j][k], ClipVal);
                }
            }
        }
    }
}

void TConvLayer::ResetGradients() {
    for (int i = 0; i < FOutputChannels; i++) {
        Filters[i]->ResetGradients();
    }
}

int TConvLayer::GetOutputChannels() {
    return FOutputChannels;
}

// ========== TPoolingLayer Class ==========
class TPoolingLayer {
private:
    int FPoolSize;
    int FStride;

    struct MaxIndex {
        int X;
        int Y;
    };

public:
    TDArray3D InputCache;
    TDArray3D OutputCache;
    std::vector<std::vector<std::vector<MaxIndex>>> MaxIndices;
    
    cl_mem InputBuffer;
    cl_mem OutputBuffer;

    TPoolingLayer(int PoolSize, int Stride)
        : FPoolSize(PoolSize),
          FStride(Stride),
          InputBuffer(nullptr),
          OutputBuffer(nullptr) {
    }
    
    ~TPoolingLayer() {
        if (InputBuffer) clReleaseMemObject(InputBuffer);
        if (OutputBuffer) clReleaseMemObject(OutputBuffer);
    }

    void Forward(const TDArray3D& Input, TDArray3D& Output) {
        InputCache = Input;

        int Channels = Input.size();
        int outH = (Input[0].size() - FPoolSize) / FStride + 1;
        int outW = (Input[0][0].size() - FPoolSize) / FStride + 1;

        Zero3DArray(Output, Channels, outH, outW);

        // Resize MaxIndices
        MaxIndices.resize(Channels);
        for (int c = 0; c < Channels; c++) {
            MaxIndices[c].resize(outH);
            for (int h = 0; h < outH; h++) {
                MaxIndices[c][h].resize(outW);
            }
        }

        for (int c = 0; c < Channels; c++) {
            for (int h = 0; h < outH; h++) {
                for (int w = 0; w < outW; w++) {
                    double MaxVal = -1e308;
                    int MaxH = 0;
                    int MaxW = 0;

                    for (int kh = 0; kh < FPoolSize; kh++) {
                        for (int kw = 0; kw < FPoolSize; kw++) {
                            int ph = h * FStride + kh;
                            int pw = w * FStride + kw;

                            if (Input[c][ph][pw] > MaxVal) {
                                MaxVal = Input[c][ph][pw];
                                MaxH = kh;
                                MaxW = kw;
                            }
                        }
                    }

                    Output[c][h][w] = MaxVal;
                    MaxIndices[c][h][w].X = MaxW;
                    MaxIndices[c][h][w].Y = MaxH;
                }
            }
        }

        OutputCache = Output;
    }

    void Backward(const TDArray3D& dOutput, TDArray3D& dInput) {
        int Channels = InputCache.size();
        int Height = InputCache[0].size();
        int Width = InputCache[0][0].size();

        Zero3DArray(dInput, Channels, Height, Width);

        for (int c = 0; c < (int)dOutput.size(); c++) {
            for (int h = 0; h < (int)dOutput[c].size(); h++) {
                for (int w = 0; w < (int)dOutput[c][h].size(); w++) {
                    int ph = h * FStride + MaxIndices[c][h][w].Y;
                    int pw = w * FStride + MaxIndices[c][h][w].X;
                    dInput[c][ph][pw] = dOutput[c][h][w];
                }
            }
        }
    }
};

// ========== TFCLayer Class (Fully Connected Layer) ==========
class TFCLayer {
private:
    int FInputSize;
    int FOutputSize;
    TActivationType FActivation;

public:
    TDArray2D W;
    DArray B;
    TDArray2D dW;
    DArray dB;
    DArray InputCache;
    DArray OutputCache;
    DArray PreActivation;
    
    cl_mem WBuffer;
    cl_mem BBuffer;
    cl_mem dWBuffer;
    cl_mem dBBuffer;

    TFCLayer(int InputSize, int OutputSize, TActivationType Activation)
    : FInputSize(InputSize),
    FOutputSize(OutputSize),
    FActivation(Activation),
    WBuffer(nullptr),
    BBuffer(nullptr),
    dWBuffer(nullptr),
    dBBuffer(nullptr) {

        double Scale;
        if (InputSize > 0) {
            Scale = std::sqrt(2.0 / InputSize);
        } else {
            Scale = 0.1;
        }

        InitMatrix(W, OutputSize, InputSize, Scale);
        ZeroArray(B, OutputSize);
        ZeroMatrix(dW, OutputSize, InputSize);
        ZeroArray(dB, OutputSize);
        
        AllocateGPUBuffers();
    }
    
    ~TFCLayer() {
        if (WBuffer) clReleaseMemObject(WBuffer);
        if (BBuffer) clReleaseMemObject(BBuffer);
        if (dWBuffer) clReleaseMemObject(dWBuffer);
        if (dBBuffer) clReleaseMemObject(dBBuffer);
    }
    
    void AllocateGPUBuffers() {
        cl_int err;
        size_t wSize = W.size() * W[0].size() * sizeof(double);
        size_t bSize = B.size() * sizeof(double);
        
        WBuffer = clCreateBuffer(gOpenCLContext.context, CL_MEM_READ_WRITE, wSize, nullptr, &err);
        BBuffer = clCreateBuffer(gOpenCLContext.context, CL_MEM_READ_WRITE, bSize, nullptr, &err);
        dWBuffer = clCreateBuffer(gOpenCLContext.context, CL_MEM_READ_WRITE, wSize, nullptr, &err);
        dBBuffer = clCreateBuffer(gOpenCLContext.context, CL_MEM_READ_WRITE, bSize, nullptr, &err);
    }
    
    void CopyToGPU() {
        std::vector<double> flatW;
        for (size_t i = 0; i < W.size(); i++) {
            for (size_t j = 0; j < W[i].size(); j++) {
                flatW.push_back(W[i][j]);
            }
        }
        clEnqueueWriteBuffer(gOpenCLContext.queue, WBuffer, CL_TRUE, 0,
                           flatW.size() * sizeof(double), flatW.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(gOpenCLContext.queue, BBuffer, CL_TRUE, 0,
                           B.size() * sizeof(double), B.data(), 0, nullptr, nullptr);
    }
    
    void CopyFromGPU() {
        std::vector<double> flatW(W.size() * W[0].size());
        clEnqueueReadBuffer(gOpenCLContext.queue, WBuffer, CL_TRUE, 0,
                          flatW.size() * sizeof(double), flatW.data(), 0, nullptr, nullptr);
        
        size_t idx = 0;
        for (size_t i = 0; i < W.size(); i++) {
            for (size_t j = 0; j < W[i].size(); j++) {
                W[i][j] = flatW[idx++];
            }
        }
        clEnqueueReadBuffer(gOpenCLContext.queue, BBuffer, CL_TRUE, 0,
                          B.size() * sizeof(double), B.data(), 0, nullptr, nullptr);
    }

    void Forward(const DArray& Input, DArray& Output) {
        InputCache = Input;
        Output.resize(FOutputSize);
        PreActivation.resize(FOutputSize);

        for (int i = 0; i < FOutputSize; i++) {
            double Sum = B[i];
            for (int j = 0; j < FInputSize; j++) {
                Sum += W[i][j] * Input[j];
            }
            PreActivation[i] = Sum;
            Output[i] = TActivation::Apply(Sum, FActivation);
        }

        OutputCache = Output;
    }

    void Backward(const DArray& dOutput, DArray& dInput) {
        dInput.resize(FInputSize);
        for (int i = 0; i < FInputSize; i++) {
            dInput[i] = 0.0;
        }

        for (int i = 0; i < FOutputSize; i++) {
            double dRaw = dOutput[i] * TActivation::Derivative(OutputCache[i], FActivation);
            dB[i] += dRaw;

            for (int j = 0; j < FInputSize; j++) {
                dW[i][j] += dRaw * InputCache[j];
                dInput[j] += dRaw * W[i][j];
            }
        }
    }

    void ApplyGradients(double LR, double ClipVal) {
        for (int i = 0; i < FOutputSize; i++) {
            B[i] -= LR * ClipValue(dB[i], ClipVal);
            dB[i] = 0.0;

            for (int j = 0; j < FInputSize; j++) {
                W[i][j] -= LR * ClipValue(dW[i][j], ClipVal);
                dW[i][j] = 0.0;
            }
        }
    }

    void ResetGradients() {
        for (int i = 0; i < FOutputSize; i++) {
            dB[i] = 0.0;
            for (int j = 0; j < FInputSize; j++) {
                dW[i][j] = 0.0;
            }
        }
    }
};

// ========== TAdvancedCNN Class ==========
class TAdvancedCNN {
private:
    int FInputWidth;
    int FInputHeight;
    int FInputChannels;
    int FOutputSize;
    TActivationType FActivation;
    TActivationType FOutputActivation;
    TLossType FLossType;
    double FLearningRate;
    double FGradientClip;

    // Store metadata for JSON serialization
    std::vector<int> FConvFilters;
    std::vector<int> FKernelSizes;
    std::vector<int> FPoolSizes;
    std::vector<int> FFCLayerSizes;

    std::vector<TConvLayer*> FConvLayers;
    std::vector<TPoolingLayer*> FPoolLayers;
    std::vector<TFCLayer*> FFullyConnectedLayers;
    TFCLayer* FOutputLayer;
    int FFlattenedSize;

    double ClipGradient(double G, double MaxVal) {
        return ClipValue(G, MaxVal);
    }

    DArray Flatten(const TDArray3D& Input) {
        int Channels = Input.size();
        int Height = Input[0].size();
        int Width = Input[0][0].size();

        DArray Result;
        Result.resize(Channels * Height * Width);

        int idx = 0;
        for (int c = 0; c < Channels; c++) {
            for (int h = 0; h < Height; h++) {
                for (int w = 0; w < Width; w++) {
                    Result[idx] = Input[c][h][w];
                    idx++;
                }
            }
        }

        return Result;
    }

    TDArray3D Unflatten(const DArray& Input, int Channels, int Height, int Width) {
        TDArray3D Result;
        Result.resize(Channels);

        int idx = 0;
        for (int c = 0; c < Channels; c++) {
            Result[c].resize(Height);
            for (int h = 0; h < Height; h++) {
                Result[c][h].resize(Width);
                for (int w = 0; w < Width; w++) {
                    Result[c][h][w] = Input[idx];
                    idx++;
                }
            }
        }

        return Result;
    }

public:
    TAdvancedCNN(int InputWidth, int InputHeight, int InputChannels,
                 const std::vector<int>& ConvFilters,
                 const std::vector<int>& KernelSizes,
                 const std::vector<int>& PoolSizes,
                 const std::vector<int>& FCLayerSizes,
                 int OutputSize,
                 TActivationType Activation,
                 TActivationType OutputActivation,
                 TLossType LossType,
                 double LearningRate,
                 double GradientClip)
         : FInputWidth(InputWidth),
           FInputHeight(InputHeight),
           FInputChannels(InputChannels),
           FOutputSize(OutputSize),
           FActivation(Activation),
           FOutputActivation(OutputActivation),
           FLossType(LossType),
           FLearningRate(LearningRate),
           FGradientClip(GradientClip),
           FConvFilters(ConvFilters),
           FKernelSizes(KernelSizes),
           FPoolSizes(PoolSizes),
           FFCLayerSizes(FCLayerSizes) {

         int CurrentChannels = InputChannels;
         int CurrentWidth = InputWidth;
         int CurrentHeight = InputHeight;

         // Create convolutional layers
         FConvLayers.resize(ConvFilters.size());
         for (size_t i = 0; i < ConvFilters.size(); i++) {
             FConvLayers[i] = new TConvLayer(CurrentChannels, ConvFilters[i],
                                             KernelSizes[i], 1, KernelSizes[i] / 2, Activation);
             CurrentChannels = ConvFilters[i];
             // CurrentWidth and CurrentHeight remain the same due to padding

             if (i < PoolSizes.size()) {
                 CurrentWidth = CurrentWidth / PoolSizes[i];
                 CurrentHeight = CurrentHeight / PoolSizes[i];
             }
         }

         // Create pooling layers
         FPoolLayers.resize(PoolSizes.size());
         for (size_t i = 0; i < PoolSizes.size(); i++) {
             FPoolLayers[i] = new TPoolingLayer(PoolSizes[i], PoolSizes[i]);
         }

         // Calculate flattened size
         FFlattenedSize = CurrentChannels * CurrentWidth * CurrentHeight;

         // Create fully connected layers
         FFullyConnectedLayers.resize(FCLayerSizes.size());
         int NumInputs = FFlattenedSize;
         for (size_t i = 0; i < FCLayerSizes.size(); i++) {
             FFullyConnectedLayers[i] = new TFCLayer(NumInputs, FCLayerSizes[i], Activation);
             NumInputs = FCLayerSizes[i];
         }

         // Create output layer
         FOutputLayer = new TFCLayer(NumInputs, OutputSize, OutputActivation);
     }

    ~TAdvancedCNN() {
        for (size_t i = 0; i < FConvLayers.size(); i++) {
            delete FConvLayers[i];
        }
        for (size_t i = 0; i < FPoolLayers.size(); i++) {
            delete FPoolLayers[i];
        }
        for (size_t i = 0; i < FFullyConnectedLayers.size(); i++) {
            delete FFullyConnectedLayers[i];
        }
        delete FOutputLayer;
    }

    // Getters and setters for properties
    double GetLearningRate() const { return FLearningRate; }
    void SetLearningRate(double value) { FLearningRate = value; }

    double GetGradientClip() const { return FGradientClip; }
    void SetGradientClip(double value) { FGradientClip = value; }

    // Methods continued in Section 8...
    DArray ForwardPass(const TDArray3D& Input);
    double BackwardPass(const DArray& Target);
    double TrainSample(const TDArray3D& Input, const DArray& Target);
    double TrainBatch(const TDArray4D& BatchInputs, const TDArray2D& BatchTargets);
    DArray Predict(const TDArray3D& Input);
    double ComputeLoss(const TDArray4D& Inputs, const TDArray2D& Targets);
    void ResetGradients();
    void ApplyGradients();

    // JSON methods continued in Section 9...
    void SaveModelToJSON(const std::string& Filename);
    void LoadModelFromJSON(const std::string& Filename);
    std::string Array1DToJSON(const DArray& Arr);
    std::string Array2DToJSON(const TDArray2D& Arr);
    std::string Array3DToJSON(const TDArray3D& Arr);
    std::string Array4DToJSON(const TDArray4D& Arr);
    
    // ONNX Export/Import
    void ExportToONNX(const std::string& Filename);
    static TAdvancedCNN* ImportFromONNX(const std::string& Filename);
    
    // Batch Normalization
    bool UseBatchNorm = false;
    std::vector<TBatchNormParams> FBatchNormParams;
    void InitializeBatchNorm();
    DArray ApplyBatchNorm(const DArray& Input, int LayerIdx);
};

// ========== TAdvancedCNN Class (continued) ==========

DArray TAdvancedCNN::ForwardPass(const TDArray3D& Input) {
    TDArray3D CurrentOutput = Input;

    // Forward through convolutional and pooling layers
    for (size_t i = 0; i < FConvLayers.size(); i++) {
        FConvLayers[i]->Forward(CurrentOutput, CurrentOutput);

        if (i < FPoolLayers.size()) {
            FPoolLayers[i]->Forward(CurrentOutput, CurrentOutput);
        }
    }

    // Flatten
    DArray FlatInput = Flatten(CurrentOutput);
    DArray LayerInput = FlatInput;

    // Forward through fully connected layers
    for (size_t i = 0; i < FFullyConnectedLayers.size(); i++) {
        FFullyConnectedLayers[i]->Forward(LayerInput, LayerInput);
    }

    // Forward through output layer
    DArray Logits;
    Logits.resize(FOutputSize);
    FOutputLayer->Forward(LayerInput, Logits);

    // Apply output activation
    if (FOutputActivation == TActivationType::atLinear) {
        return Logits;
    } else {
        TActivation::ApplySoftmax(Logits);
        return Logits;
    }
}

double TAdvancedCNN::BackwardPass(const DArray& Target) {
    // Compute output gradient
    DArray OutputGrad;
    OutputGrad.resize(FOutputSize);
    for (int i = 0; i < FOutputSize; i++) {
        OutputGrad[i] = FOutputLayer->OutputCache[i] - Target[i];
    }

    // Backward through output layer
    DArray FCGrad;
    FOutputLayer->Backward(OutputGrad, FCGrad);

    // Backward through fully connected layers
    for (int i = FFullyConnectedLayers.size() - 1; i >= 0; i--) {
        FFullyConnectedLayers[i]->Backward(FCGrad, FCGrad);
    }

    // Unflatten gradient
    int LastConvIdx = FConvLayers.size() - 1;
    TDArray3D CurrentGrad = Unflatten(FCGrad,
                                      FConvLayers[LastConvIdx]->OutputCache.size(),
                                      FConvLayers[LastConvIdx]->OutputCache[0].size(),
                                      FConvLayers[LastConvIdx]->OutputCache[0][0].size());

    // Backward through convolutional and pooling layers
    for (int i = FConvLayers.size() - 1; i >= 0; i--) {
        if (i < (int)FPoolLayers.size()) {
            FPoolLayers[i]->Backward(CurrentGrad, CurrentGrad);
        }
        FConvLayers[i]->Backward(CurrentGrad, CurrentGrad);
    }

    // Compute and return loss
    return TLoss::Compute(FOutputLayer->OutputCache, Target, FLossType);
}

double TAdvancedCNN::TrainSample(const TDArray3D& Input, const DArray& Target) {
    ResetGradients();
    ForwardPass(Input);
    double Loss = BackwardPass(Target);
    ApplyGradients();
    return Loss;
}

double TAdvancedCNN::TrainBatch(const TDArray4D& BatchInputs, const TDArray2D& BatchTargets) {
    ResetGradients();
    double BatchLoss = 0.0;

    for (size_t b = 0; b < BatchInputs.size(); b++) {
        ForwardPass(BatchInputs[b]);
        BatchLoss += BackwardPass(BatchTargets[b]);
    }

    ApplyGradients();
    return BatchLoss / BatchInputs.size();
}

DArray TAdvancedCNN::Predict(const TDArray3D& Input) {
    return ForwardPass(Input);
}

double TAdvancedCNN::ComputeLoss(const TDArray4D& Inputs, const TDArray2D& Targets) {
    double Result = 0.0;

    for (size_t i = 0; i < Inputs.size(); i++) {
        DArray Output = ForwardPass(Inputs[i]);
        Result += TLoss::Compute(Output, Targets[i], FLossType);
    }

    return Result / Inputs.size();
}

void TAdvancedCNN::ResetGradients() {
    for (size_t i = 0; i < FConvLayers.size(); i++) {
        FConvLayers[i]->ResetGradients();
    }
    for (size_t i = 0; i < FFullyConnectedLayers.size(); i++) {
        FFullyConnectedLayers[i]->ResetGradients();
    }
    FOutputLayer->ResetGradients();
}

// ========== TAdvancedCNN Class (continued from Section 8) ==========

void TAdvancedCNN::ApplyGradients() {
    for (size_t i = 0; i < FConvLayers.size(); i++) {
        FConvLayers[i]->ApplyGradients(FLearningRate, FGradientClip);
    }
    for (size_t i = 0; i < FFullyConnectedLayers.size(); i++) {
        FFullyConnectedLayers[i]->ApplyGradients(FLearningRate, FGradientClip);
    }
    FOutputLayer->ApplyGradients(FLearningRate, FGradientClip);
}

// JSON Serialization Helper Functions
std::string TAdvancedCNN::Array1DToJSON(const DArray& Arr) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "[";
    for (size_t i = 0; i < Arr.size(); i++) {
        oss << Arr[i];
        if (i < Arr.size() - 1) oss << ",";
    }
    oss << "]";
    return oss.str();
}

std::string TAdvancedCNN::Array2DToJSON(const TDArray2D& Arr) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "[";
    for (size_t i = 0; i < Arr.size(); i++) {
        oss << Array1DToJSON(Arr[i]);
        if (i < Arr.size() - 1) oss << ",";
    }
    oss << "]";
    return oss.str();
}

std::string TAdvancedCNN::Array3DToJSON(const TDArray3D& Arr) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "[";
    for (size_t i = 0; i < Arr.size(); i++) {
        oss << Array2DToJSON(Arr[i]);
        if (i < Arr.size() - 1) oss << ",";
    }
    oss << "]";
    return oss.str();
}

std::string TAdvancedCNN::Array4DToJSON(const TDArray4D& Arr) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "[";
    for (size_t i = 0; i < Arr.size(); i++) {
        oss << Array3DToJSON(Arr[i]);
        if (i < Arr.size() - 1) oss << ",";
    }
    oss << "]";
    return oss.str();
}

void TAdvancedCNN::SaveModelToJSON(const std::string& Filename) {
    std::ofstream file(Filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + Filename);
    }

    file << std::setprecision(17);
    file << "{\n";

    // Model architecture
    file << "  \"input_width\": " << FInputWidth << ",\n";
    file << "  \"input_height\": " << FInputHeight << ",\n";
    file << "  \"input_channels\": " << FInputChannels << ",\n";
    file << "  \"output_size\": " << FOutputSize << ",\n";
    file << "  \"learning_rate\": " << FLearningRate << ",\n";
    file << "  \"gradient_clip\": " << FGradientClip << ",\n";

    // Activation types
    file << "  \"activation\": " << static_cast<int>(FActivation) << ",\n";
    file << "  \"output_activation\": " << static_cast<int>(FOutputActivation) << ",\n";
    file << "  \"loss_type\": " << static_cast<int>(FLossType) << ",\n";

    // Layer metadata - required for cross-app compatibility
    file << "  \"conv_filters\": [";
    for (size_t i = 0; i < FConvFilters.size(); i++) {
        if (i > 0) file << ",";
        file << FConvFilters[i];
    }
    file << "],\n";

    file << "  \"kernel_sizes\": [";
    for (size_t i = 0; i < FKernelSizes.size(); i++) {
        if (i > 0) file << ",";
        file << FKernelSizes[i];
    }
    file << "],\n";

    file << "  \"pool_sizes\": [";
    for (size_t i = 0; i < FPoolSizes.size(); i++) {
        if (i > 0) file << ",";
        file << FPoolSizes[i];
    }
    file << "],\n";

    file << "  \"fc_layer_sizes\": [";
    for (size_t i = 0; i < FFCLayerSizes.size(); i++) {
        if (i > 0) file << ",";
        file << FFCLayerSizes[i];
    }
    file << "],\n";

    // Convolutional layers
    file << "  \"conv_layers\": [\n";
    for (size_t i = 0; i < FConvLayers.size(); i++) {
        file << "    {\n";
        file << "      \"filters\": [\n";
        for (size_t f = 0; f < FConvLayers[i]->Filters.size(); f++) {
            file << "        {\n";
            file << "          \"weights\": " << Array4DToJSON(FConvLayers[i]->Filters[f]->Weights) << ",\n";
            file << "          \"bias\": " << FConvLayers[i]->Filters[f]->Bias << "\n";
            file << "        }";
            if (f < FConvLayers[i]->Filters.size() - 1) file << ",";
            file << "\n";
        }
        file << "      ]\n";
        file << "    }";
        if (i < FConvLayers.size() - 1) file << ",";
        file << "\n";
    }
    file << "  ],\n";

    // Fully connected layers
    file << "  \"fc_layers\": [\n";
    for (size_t i = 0; i < FFullyConnectedLayers.size(); i++) {
        file << "    {\n";
        file << "      \"weights\": " << Array2DToJSON(FFullyConnectedLayers[i]->W) << ",\n";
        file << "      \"bias\": " << Array1DToJSON(FFullyConnectedLayers[i]->B) << "\n";
        file << "    }";
        if (i < FFullyConnectedLayers.size() - 1) file << ",";
        file << "\n";
    }
    file << "  ],\n";

    // Output layer
    file << "  \"output_layer\": {\n";
    file << "    \"weights\": " << Array2DToJSON(FOutputLayer->W) << ",\n";
    file << "    \"bias\": " << Array1DToJSON(FOutputLayer->B) << "\n";
    file << "  }\n";

    file << "}\n";
    file.close();
}

void TAdvancedCNN::LoadModelFromJSON(const std::string& Filename) {
    std::ifstream file(Filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + Filename);
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();

    // Simple JSON parser for the specific structure we saved
    // Note: This is a basic implementation.  For production use, consider a proper JSON library.

    auto findValue = [&content](const std::string& key) -> std::string {
        std::string searchKey = "\"" + key + "\": ";
        size_t pos = content.find(searchKey);
        if (pos == std::string::npos) return "";

        pos += searchKey.length();
        while (pos < content.length() && (content[pos] == ' ' || content[pos] == '\n')) pos++;

        size_t endPos = pos;
        if (content[pos] == '"') {
            endPos = content.find('"', pos + 1);
            return content.substr(pos + 1, endPos - pos - 1);
        } else if (content[pos] == '[' || content[pos] == '{') {
            int depth = 0;
            char startChar = content[pos];
            char endChar = (startChar == '[') ? ']' : '}';
            do {
                if (content[endPos] == startChar) depth++;
                if (content[endPos] == endChar) depth--;
                endPos++;
            } while (depth > 0 && endPos < content.length());
            return content.substr(pos, endPos - pos);
        } else {
            while (endPos < content.length() && content[endPos] != ',' &&
                   content[endPos] != '\n' && content[endPos] != '}') endPos++;
            return content.substr(pos, endPos - pos);
        }
    };

    auto parseArray1D = [](const std::string& str) -> DArray {
        DArray result;
        if (str.empty() || str[0] != '[') return result;

        std::string nums = str.substr(1, str.length() - 2);
        std::istringstream iss(nums);
        std::string token;
        while (std::getline(iss, token, ',')) {
            result.push_back(std::stod(token));
        }
        return result;
    };

    auto parseArray2D = [&parseArray1D](const std::string& str) -> TDArray2D {
        TDArray2D result;
        if (str.empty() || str[0] != '[') return result;

        size_t pos = 1;
        int depth = 0;
        size_t start = pos;

        while (pos < str.length() - 1) {
            if (str[pos] == '[') {
                if (depth == 0) start = pos;
                depth++;
            } else if (str[pos] == ']') {
                depth--;
                if (depth == 0) {
                    result.push_back(parseArray1D(str.substr(start, pos - start + 1)));
                }
            }
            pos++;
        }
        return result;
    };

    auto parseArray4D = [&](const std::string& str) -> TDArray4D {
        TDArray4D result;
        if (str.empty() || str[0] != '[') return result;

        size_t pos = 1;
        int depth = 0;
        size_t start = pos;

        while (pos < str.length() - 1) {
            if (str[pos] == '[') {
                if (depth == 0) start = pos;
                depth++;
            } else if (str[pos] == ']') {
                depth--;
                if (depth == 0) {
                    // Parse 3D array
                    TDArray3D arr3d;
                    std::string str3d = str.substr(start, pos - start + 1);

                    size_t pos3 = 1;
                    int depth3 = 0;
                    size_t start3 = pos3;

                    while (pos3 < str3d.length() - 1) {
                        if (str3d[pos3] == '[') {
                            if (depth3 == 0) start3 = pos3;
                            depth3++;
                        } else if (str3d[pos3] == ']') {
                            depth3--;
                            if (depth3 == 0) {
                                arr3d.push_back(parseArray2D(str3d.substr(start3, pos3 - start3 + 1)));
                            }
                        }
                        pos3++;
                    }
                    result.push_back(arr3d);
                }
            }
            pos++;
        }
        return result;
    };

    // Load architecture parameters
    FLearningRate = std::stod(findValue("learning_rate"));
    FGradientClip = std::stod(findValue("gradient_clip"));

    // Load convolutional layers weights
    std::string convLayersStr = findValue("conv_layers");
    size_t convPos = 0;
    int convDepth = 0;
    size_t convLayerIdx = 0;

    while (convPos < convLayersStr.length() && convLayerIdx < FConvLayers.size()) {
        if (convLayersStr[convPos] == '{' && convDepth == 1) {
            // Find filters array for this layer
            size_t filtersStart = convLayersStr.find("\"filters\":", convPos);
            if (filtersStart != std::string::npos) {
                filtersStart = convLayersStr.find('[', filtersStart);
                int filterDepth = 0;
                size_t filtersEnd = filtersStart;
                do {
                    if (convLayersStr[filtersEnd] == '[') filterDepth++;
                    if (convLayersStr[filtersEnd] == ']') filterDepth--;
                    filtersEnd++;
                } while (filterDepth > 0);

                std::string filtersStr = convLayersStr.substr(filtersStart, filtersEnd - filtersStart);

                // Parse each filter
                size_t filterPos = 1;
                size_t filterIdx = 0;
                int fDepth = 0;

                while (filterPos < filtersStr.length() && filterIdx < FConvLayers[convLayerIdx]->Filters.size()) {
                    if (filtersStr[filterPos] == '{') {
                        if (fDepth == 0) {
                            size_t weightsStart = filtersStr.find("\"weights\":", filterPos);
                            size_t weightsValStart = filtersStr.find('[', weightsStart);
                            int wDepth = 0;
                            size_t weightsEnd = weightsValStart;
                            do {
                                if (filtersStr[weightsEnd] == '[') wDepth++;
                                if (filtersStr[weightsEnd] == ']') wDepth--;
                                weightsEnd++;
                            } while (wDepth > 0);

                            std::string weightsStr = filtersStr.substr(weightsValStart, weightsEnd - weightsValStart);
                            FConvLayers[convLayerIdx]->Filters[filterIdx]->Weights = parseArray4D(weightsStr);

                            size_t biasStart = filtersStr.find("\"bias\":", filterPos);
                            size_t biasValStart = biasStart + 7;
                            while (filtersStr[biasValStart] == ' ') biasValStart++;
                            size_t biasEnd = filtersStr.find_first_of(",\n}", biasValStart);
                            FConvLayers[convLayerIdx]->Filters[filterIdx]->Bias =
                                std::stod(filtersStr.substr(biasValStart, biasEnd - biasValStart));

                            filterIdx++;
                        }
                        fDepth++;
                    } else if (filtersStr[filterPos] == '}') {
                        fDepth--;
                    }
                    filterPos++;
                }

                convPos = filtersEnd + convPos;
                convLayerIdx++;
            }
        }
        if (convLayersStr[convPos] == '{') convDepth++;
        if (convLayersStr[convPos] == '}') convDepth--;
        convPos++;
    }

    // Load fully connected layers
    std::string fcLayersStr = findValue("fc_layers");
    size_t fcPos = 1;
    size_t fcLayerIdx = 0;
    int fcDepth = 0;

    while (fcPos < fcLayersStr.length() && fcLayerIdx < FFullyConnectedLayers.size()) {
        if (fcLayersStr[fcPos] == '{') {
            if (fcDepth == 0) {
                size_t weightsStart = fcLayersStr.find("\"weights\":", fcPos);
                size_t weightsValStart = fcLayersStr.find('[', weightsStart);
                int wDepth = 0;
                size_t weightsEnd = weightsValStart;
                do {
                    if (fcLayersStr[weightsEnd] == '[') wDepth++;
                    if (fcLayersStr[weightsEnd] == ']') wDepth--;
                    weightsEnd++;
                } while (wDepth > 0);

                std::string weightsStr = fcLayersStr.substr(weightsValStart, weightsEnd - weightsValStart);
                FFullyConnectedLayers[fcLayerIdx]->W = parseArray2D(weightsStr);

                size_t biasStart = fcLayersStr.find("\"bias\":", fcPos);
                size_t biasValStart = fcLayersStr.find('[', biasStart);
                int bDepth = 0;
                size_t biasEnd = biasValStart;
                do {
                    if (fcLayersStr[biasEnd] == '[') bDepth++;
                    if (fcLayersStr[biasEnd] == ']') bDepth--;
                    biasEnd++;
                } while (bDepth > 0);

                std::string biasStr = fcLayersStr.substr(biasValStart, biasEnd - biasValStart);
                FFullyConnectedLayers[fcLayerIdx]->B = parseArray1D(biasStr);

                fcLayerIdx++;
            }
            fcDepth++;
        } else if (fcLayersStr[fcPos] == '}') {
            fcDepth--;
        }
        fcPos++;
    }

    // Load output layer
    std::string outputLayerStr = findValue("output_layer");

    size_t weightsStart = outputLayerStr.find("\"weights\":");
    size_t weightsValStart = outputLayerStr.find('[', weightsStart);
    int wDepth = 0;
    size_t weightsEnd = weightsValStart;
    do {
        if (outputLayerStr[weightsEnd] == '[') wDepth++;
        if (outputLayerStr[weightsEnd] == ']') wDepth--;
        weightsEnd++;
    } while (wDepth > 0);

    std::string weightsStr = outputLayerStr.substr(weightsValStart, weightsEnd - weightsValStart);
    FOutputLayer->W = parseArray2D(weightsStr);

    size_t biasStart = outputLayerStr.find("\"bias\":");
    size_t biasValStart = outputLayerStr.find('[', biasStart);
    int bDepth = 0;
    size_t biasEnd = biasValStart;
    do {
        if (outputLayerStr[biasEnd] == '[') bDepth++;
        if (outputLayerStr[biasEnd] == ']') bDepth--;
        biasEnd++;
    } while (bDepth > 0);

    std::string biasStr = outputLayerStr.substr(biasValStart, biasEnd - biasValStart);
    FOutputLayer->B = parseArray1D(biasStr);
}

// ========== Batch Normalization Methods ==========

void TAdvancedCNN::InitializeBatchNorm() {
    FBatchNormParams.clear();
    for (size_t i = 0; i < FConvLayers.size(); i++) {
        TBatchNormParams params;
        params.Initialize(FConvFilters[i]);
        FBatchNormParams.push_back(params);
    }
}

DArray TAdvancedCNN::ApplyBatchNorm(const DArray& Input, int LayerIdx) {
    if (!UseBatchNorm || LayerIdx >= (int)FBatchNormParams.size()) {
        return Input;
    }
    
    const TBatchNormParams& params = FBatchNormParams[LayerIdx];
    DArray Output(Input.size());
    
    int channelSize = Input.size() / params.Gamma.size();
    
    for (size_t c = 0; c < params.Gamma.size(); c++) {
        for (int i = 0; i < channelSize; i++) {
            int idx = c * channelSize + i;
            if (idx < (int)Input.size()) {
                double normalized = (Input[idx] - params.RunningMean[c]) / 
                                   std::sqrt(params.RunningVar[c] + params.Epsilon);
                Output[idx] = params.Gamma[c] * normalized + params.Beta[c];
            }
        }
    }
    
    return Output;
}

// ========== ONNX Export/Import Methods ==========

void TAdvancedCNN::ExportToONNX(const std::string& Filename) {
    std::ofstream file(Filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + Filename);
    }
    
    // Write ONNX magic header
    const char magic[] = "ONNX";
    file.write(magic, 4);
    
    // Write version
    int version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(int));
    
    // Write model metadata
    file.write(reinterpret_cast<const char*>(&FInputWidth), sizeof(int));
    file.write(reinterpret_cast<const char*>(&FInputHeight), sizeof(int));
    file.write(reinterpret_cast<const char*>(&FInputChannels), sizeof(int));
    file.write(reinterpret_cast<const char*>(&FOutputSize), sizeof(int));
    
    int useBN = UseBatchNorm ? 1 : 0;
    file.write(reinterpret_cast<const char*>(&useBN), sizeof(int));
    
    // Write conv layer count and metadata
    int numConvLayers = FConvLayers.size();
    file.write(reinterpret_cast<const char*>(&numConvLayers), sizeof(int));
    
    for (int i = 0; i < numConvLayers; i++) {
        file.write(reinterpret_cast<const char*>(&FConvFilters[i]), sizeof(int));
        file.write(reinterpret_cast<const char*>(&FKernelSizes[i]), sizeof(int));
        file.write(reinterpret_cast<const char*>(&FPoolSizes[i]), sizeof(int));
    }
    
    // Write FC layer count and sizes
    int numFCLayers = FFCLayerSizes.size();
    file.write(reinterpret_cast<const char*>(&numFCLayers), sizeof(int));
    for (int i = 0; i < numFCLayers; i++) {
        file.write(reinterpret_cast<const char*>(&FFCLayerSizes[i]), sizeof(int));
    }
    
    // Write conv layer weights
    for (size_t i = 0; i < FConvLayers.size(); i++) {
        TConvLayer* layer = FConvLayers[i];
        int numFilters = layer->Filters.size();
        file.write(reinterpret_cast<const char*>(&numFilters), sizeof(int));
        
        for (int f = 0; f < numFilters; f++) {
            TConvFilter* filter = layer->Filters[f];
            // Write filter weights (4D: OutputChannels x InputChannels x KernelH x KernelW)
            int d0 = filter->Weights.size();
            int d1 = d0 > 0 ? filter->Weights[0].size() : 0;
            int d2 = d1 > 0 ? filter->Weights[0][0].size() : 0;
            int d3 = d2 > 0 ? filter->Weights[0][0][0].size() : 0;
            
            file.write(reinterpret_cast<const char*>(&d0), sizeof(int));
            file.write(reinterpret_cast<const char*>(&d1), sizeof(int));
            file.write(reinterpret_cast<const char*>(&d2), sizeof(int));
            file.write(reinterpret_cast<const char*>(&d3), sizeof(int));
            
            for (int a = 0; a < d0; a++) {
                for (int b = 0; b < d1; b++) {
                    for (int c = 0; c < d2; c++) {
                        for (int d = 0; d < d3; d++) {
                            file.write(reinterpret_cast<const char*>(&filter->Weights[a][b][c][d]), sizeof(double));
                        }
                    }
                }
            }
            
            // Write bias
            file.write(reinterpret_cast<const char*>(&filter->Bias), sizeof(double));
        }
    }
    
    // Write FC layer weights
    for (size_t i = 0; i < FFullyConnectedLayers.size(); i++) {
        TFCLayer* layer = FFullyConnectedLayers[i];
        int rows = layer->W.size();
        int cols = rows > 0 ? layer->W[0].size() : 0;
        
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                file.write(reinterpret_cast<const char*>(&layer->W[r][c]), sizeof(double));
            }
        }
        
        int biasSize = layer->B.size();
        file.write(reinterpret_cast<const char*>(&biasSize), sizeof(int));
        for (int b = 0; b < biasSize; b++) {
            file.write(reinterpret_cast<const char*>(&layer->B[b]), sizeof(double));
        }
    }
    
    // Write output layer
    int outRows = FOutputLayer->W.size();
    int outCols = outRows > 0 ? FOutputLayer->W[0].size() : 0;
    file.write(reinterpret_cast<const char*>(&outRows), sizeof(int));
    file.write(reinterpret_cast<const char*>(&outCols), sizeof(int));
    
    for (int r = 0; r < outRows; r++) {
        for (int c = 0; c < outCols; c++) {
            file.write(reinterpret_cast<const char*>(&FOutputLayer->W[r][c]), sizeof(double));
        }
    }
    
    int outBiasSize = FOutputLayer->B.size();
    file.write(reinterpret_cast<const char*>(&outBiasSize), sizeof(int));
    for (int b = 0; b < outBiasSize; b++) {
        file.write(reinterpret_cast<const char*>(&FOutputLayer->B[b]), sizeof(double));
    }
    
    // Write batch norm params if enabled
    if (UseBatchNorm) {
        for (size_t i = 0; i < FBatchNormParams.size(); i++) {
            const TBatchNormParams& bn = FBatchNormParams[i];
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
    std::cout << "Model exported to ONNX: " << Filename << std::endl;
}

TAdvancedCNN* TAdvancedCNN::ImportFromONNX(const std::string& Filename) {
    std::ifstream file(Filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + Filename);
    }
    
    // Read and verify magic header
    char magic[5] = {0};
    file.read(magic, 4);
    if (std::string(magic) != "ONNX") {
        throw std::runtime_error("Invalid ONNX file format");
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
    
    std::vector<int> convFilters(numConvLayers);
    std::vector<int> kernelSizes(numConvLayers);
    std::vector<int> poolSizes(numConvLayers);
    
    for (int i = 0; i < numConvLayers; i++) {
        file.read(reinterpret_cast<char*>(&convFilters[i]), sizeof(int));
        file.read(reinterpret_cast<char*>(&kernelSizes[i]), sizeof(int));
        file.read(reinterpret_cast<char*>(&poolSizes[i]), sizeof(int));
    }
    
    // Read FC layer sizes
    int numFCLayers;
    file.read(reinterpret_cast<char*>(&numFCLayers), sizeof(int));
    std::vector<int> fcSizes(numFCLayers);
    for (int i = 0; i < numFCLayers; i++) {
        file.read(reinterpret_cast<char*>(&fcSizes[i]), sizeof(int));
    }
    
    // Create CNN with loaded architecture
    TAdvancedCNN* cnn = new TAdvancedCNN(inputW, inputH, inputC,
                                          convFilters, kernelSizes, poolSizes,
                                          fcSizes, outputSize,
                                          TActivationType::atReLU,
                                          TActivationType::atLinear,
                                          TLossType::ltCrossEntropy,
                                          0.001, 5.0);
    cnn->UseBatchNorm = (useBN == 1);
    
    // Read conv layer weights
    for (int i = 0; i < numConvLayers; i++) {
        int numFilters;
        file.read(reinterpret_cast<char*>(&numFilters), sizeof(int));
        
        for (int f = 0; f < numFilters; f++) {
            int d0, d1, d2, d3;
            file.read(reinterpret_cast<char*>(&d0), sizeof(int));
            file.read(reinterpret_cast<char*>(&d1), sizeof(int));
            file.read(reinterpret_cast<char*>(&d2), sizeof(int));
            file.read(reinterpret_cast<char*>(&d3), sizeof(int));
            
            TConvFilter* filter = cnn->FConvLayers[i]->Filters[f];
            filter->Weights.resize(d0);
            for (int a = 0; a < d0; a++) {
                filter->Weights[a].resize(d1);
                for (int b = 0; b < d1; b++) {
                    filter->Weights[a][b].resize(d2);
                    for (int c = 0; c < d2; c++) {
                        filter->Weights[a][b][c].resize(d3);
                        for (int d = 0; d < d3; d++) {
                            file.read(reinterpret_cast<char*>(&filter->Weights[a][b][c][d]), sizeof(double));
                        }
                    }
                }
            }
            
            file.read(reinterpret_cast<char*>(&filter->Bias), sizeof(double));
        }
    }
    
    // Read FC layer weights
    for (int i = 0; i < (int)cnn->FFullyConnectedLayers.size(); i++) {
        int rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        
        cnn->FFullyConnectedLayers[i]->W.resize(rows);
        for (int r = 0; r < rows; r++) {
            cnn->FFullyConnectedLayers[i]->W[r].resize(cols);
            for (int c = 0; c < cols; c++) {
                file.read(reinterpret_cast<char*>(&cnn->FFullyConnectedLayers[i]->W[r][c]), sizeof(double));
            }
        }
        
        int biasSize;
        file.read(reinterpret_cast<char*>(&biasSize), sizeof(int));
        cnn->FFullyConnectedLayers[i]->B.resize(biasSize);
        for (int b = 0; b < biasSize; b++) {
            file.read(reinterpret_cast<char*>(&cnn->FFullyConnectedLayers[i]->B[b]), sizeof(double));
        }
    }
    
    // Read output layer
    int outRows, outCols;
    file.read(reinterpret_cast<char*>(&outRows), sizeof(int));
    file.read(reinterpret_cast<char*>(&outCols), sizeof(int));
    
    cnn->FOutputLayer->W.resize(outRows);
    for (int r = 0; r < outRows; r++) {
        cnn->FOutputLayer->W[r].resize(outCols);
        for (int c = 0; c < outCols; c++) {
            file.read(reinterpret_cast<char*>(&cnn->FOutputLayer->W[r][c]), sizeof(double));
        }
    }
    
    int outBiasSize;
    file.read(reinterpret_cast<char*>(&outBiasSize), sizeof(int));
    cnn->FOutputLayer->B.resize(outBiasSize);
    for (int b = 0; b < outBiasSize; b++) {
        file.read(reinterpret_cast<char*>(&cnn->FOutputLayer->B[b]), sizeof(double));
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

// ========== CLI Helper Functions ==========

void PrintHelp() {
    std::cout << "Commands:\n";
    std::cout << "  create      Create a new CNN model and save to JSON\n";
    std::cout << "  train       Train an existing model with data from JSON\n";
    std::cout << "  predict     Make predictions with a trained model from JSON\n";
    std::cout << "  info        Display model information from JSON\n";
    std::cout << "  export-onnx Export model to ONNX format\n";
    std::cout << "  import-onnx Import model from ONNX format\n";
    std::cout << "  help        Show this help message\n\n";
    std::cout << "Create Options:\n";
    std::cout << "  --input-w=N            Input width (required)\n";
    std::cout << "  --input-h=N            Input height (required)\n";
    std::cout << "  --input-c=N            Input channels (required)\n";
    std::cout << "  --conv=N,N,...         Conv filters (required)\n";
    std::cout << "  --kernels=N,N,...      Kernel sizes (required)\n";
    std::cout << "  --pools=N,N,...        Pool sizes (required)\n";
    std::cout << "  --fc=N,N,...           FC layer sizes (required)\n";
    std::cout << "  --output=N             Output layer size (required)\n";
    std::cout << "  --save=FILE.json       Save model to JSON file (required)\n";
    std::cout << "  --lr=VALUE             Learning rate (default: 0.001)\n";
    std::cout << "  --hidden-act=TYPE      sigmoid|tanh|relu|linear (default: relu)\n";
    std::cout << "  --output-act=TYPE      sigmoid|tanh|relu|linear (default: linear)\n";
    std::cout << "  --loss=TYPE            mse|crossentropy (default: mse)\n";
    std::cout << "  --clip=VALUE           Gradient clipping (default: 5.0)\n";
    std::cout << "  --batch-norm           Enable batch normalization\n\n";
    std::cout << "Train Options:\n";
    std::cout << "  --model=FILE.json      Load model from JSON file (required)\n";
    std::cout << "  --data=FILE.csv        Training data CSV file (required)\n";
    std::cout << "  --epochs=N             Number of epochs (required)\n";
    std::cout << "  --save=FILE.json       Save trained model to JSON (required)\n";
    std::cout << "  --batch-size=N         Batch size (default: 32)\n\n";
    std::cout << "Predict Options:\n";
    std::cout << "  --model=FILE.json      Load model from JSON file (required)\n";
    std::cout << "  --data=FILE.csv        Input data CSV file (required)\n";
    std::cout << "  --output=FILE.csv      Save predictions to CSV file (required)\n\n";
    std::cout << "Info Options:\n";
    std::cout << "  --model=FILE.json      Load model from JSON file (required)\n\n";
    std::cout << "Export ONNX Options:\n";
    std::cout << "  --model=FILE.json      Load model from JSON file (required)\n";
    std::cout << "  --onnx=FILE.onnx       Save to ONNX file (required)\n\n";
    std::cout << "Import ONNX Options:\n";
    std::cout << "  --onnx=FILE.onnx       Load from ONNX file (required)\n";
    std::cout << "  --save=FILE.json       Save to JSON file (required)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  cnn create --input-w=28 --input-h=28 --input-c=1 --conv=32,64 --kernels=3,3 --pools=2,2 --fc=128 --output=10 --save=model.json\n";
    std::cout << "  cnn create --input-w=28 --input-h=28 --input-c=1 --conv=32,64 --kernels=3,3 --pools=2,2 --fc=128 --output=10 --batch-norm --save=model.json\n";
    std::cout << "  cnn train --model=model.json --data=data.csv --epochs=50 --save=model_trained.json\n";
    std::cout << "  cnn predict --model=model_trained.json --data=test.csv --output=predictions.csv\n";
    std::cout << "  cnn info --model=model.json\n";
    std::cout << "  cnn export-onnx --model=model.json --onnx=model.onnx\n";
    std::cout << "  cnn import-onnx --onnx=model.onnx --save=imported.json\n";
}

void PrintModelInfo(const std::string& modelFile) {
    std::ifstream file(modelFile);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open model file: " << modelFile << std::endl;
        return;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();

    auto findValue = [&content](const std::string& key) -> std::string {
        std::string searchKey = "\"" + key + "\": ";
        size_t pos = content.find(searchKey);
        if (pos == std::string::npos) return "";
        pos += searchKey.length();
        while (pos < content.length() && (content[pos] == ' ' || content[pos] == '\n')) pos++;
        size_t endPos = pos;
        while (endPos < content.length() && content[endPos] != ',' &&
            content[endPos] != '\n' && content[endPos] != '}') endPos++;
        return content.substr(pos, endPos - pos);
    };

    std::cout << "\n=================================================================\n";
    std::cout << "  Model Information:  " << modelFile << "\n";
    std::cout << "=================================================================\n\n";
    std::cout << "Architecture:\n";
    std::cout << "Input: " << findValue("input_width") << "x"
    << findValue("input_height") << "x"
    << findValue("input_channels") << "\n";
    std::cout << "Output size: " << findValue("output_size") << "\n\n";
    std::cout << "Training Parameters:\n";
    std::cout << "Learning rate: " << findValue("learning_rate") << "\n";
    std::cout << "Gradient clip: " << findValue("gradient_clip") << "\n";
    std::cout << "activation: " << findValue("activation") << "\n";
    std::cout << "output_activation: " << findValue("output_activation") << "\n";
    std::cout << "loss_type: " << findValue("loss_type") << "\n\n";
}

std::vector<int> ParseIntList(const std::string& str) {
    std::vector<int> result;
    std::istringstream iss(str);
    std::string token;
    while (std::getline(iss, token, ',')) {
        result.push_back(std::stoi(token));
    }
    return result;
}

std::string GetArgValue(int argc, char* argv[], const std::string& arg, const std::string& defaultValue = "") {
    // Handle both --key=value and --key value formats
    std::string searchKey = arg + "=";
    
    for (int i = 1; i < argc; i++) {
        std::string argv_str(argv[i]);
        
        // Check for --key=value format
        if (argv_str.substr(0, searchKey.length()) == searchKey) {
            return argv_str.substr(searchKey.length());
        }
        
        // Check for --key value format
        if (argv_str == arg && i + 1 < argc) {
            return std::string(argv[i + 1]);
        }
    }
    return defaultValue;
}

bool HasArg(int argc, char* argv[], const std::string& arg) {
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == arg) {
            return true;
        }
    }
    return false;
}

TActivationType ParseActivationType(const std::string& str) {
    if (str == "sigmoid") return TActivationType::atSigmoid;
    if (str == "tanh") return TActivationType::atTanh;
    if (str == "relu") return TActivationType::atReLU;
    if (str == "linear") return TActivationType::atLinear;
    return TActivationType::atReLU;  // default
}

TLossType ParseLossType(const std::string& str) {
    if (str == "mse") return TLossType::ltMSE;
    if (str == "crossentropy") return TLossType::ltCrossEntropy;
    return TLossType::ltCrossEntropy;  // default
}

TCommand ParseCommand(const std::string& cmd) {
    if (cmd == "help") return TCommand::cmdHelp;
    if (cmd == "info") return TCommand::cmdInfo;
    if (cmd == "create") return TCommand::cmdCreate;
    if (cmd == "train") return TCommand::cmdTrain;
    if (cmd == "predict") return TCommand::cmdPredict;
    if (cmd == "export-onnx") return TCommand::cmdExportONNX;
    if (cmd == "import-onnx") return TCommand::cmdImportONNX;
    return TCommand::cmdNone;
}

std::string ActivationTypeToStr(TActivationType act) {
    switch (act) {
        case TActivationType::atSigmoid: return "sigmoid";
        case TActivationType::atTanh: return "tanh";
        case TActivationType::atReLU: return "relu";
        case TActivationType::atLinear: return "linear";
        default: return "unknown";
    }
}

std::string LossTypeToStr(TLossType loss) {
    switch (loss) {
        case TLossType::ltMSE: return "mse";
        case TLossType::ltCrossEntropy: return "crossentropy";
        default: return "unknown";
    }
}

void HandleCreate(int argc, char* argv[]) {
     std::string saveFile = GetArgValue(argc, argv, "--save", "");
     std::string inputWStr = GetArgValue(argc, argv, "--input-w", "");
     std::string inputHStr = GetArgValue(argc, argv, "--input-h", "");
     std::string inputCStr = GetArgValue(argc, argv, "--input-c", "");
     std::string convFilters = GetArgValue(argc, argv, "--conv", "");
     std::string kernels = GetArgValue(argc, argv, "--kernels", "");
     std::string pools = GetArgValue(argc, argv, "--pools", "");
     std::string fcLayers = GetArgValue(argc, argv, "--fc", "");
     std::string outputSizeStr = GetArgValue(argc, argv, "--output", "");

     // Validate required arguments
     if (saveFile.empty()) {
         std::cerr << "Error: --save argument is required for create command" << std::endl;
         return;
     }
     if (inputWStr.empty()) {
         std::cerr << "Error: --input-w argument is required for create command" << std::endl;
         return;
     }
     if (inputHStr.empty()) {
         std::cerr << "Error: --input-h argument is required for create command" << std::endl;
         return;
     }
     if (inputCStr.empty()) {
         std::cerr << "Error: --input-c argument is required for create command" << std::endl;
         return;
     }
     if (convFilters.empty()) {
         std::cerr << "Error: --conv argument is required for create command" << std::endl;
         return;
     }
     if (kernels.empty()) {
         std::cerr << "Error: --kernels argument is required for create command" << std::endl;
         return;
     }
     if (pools.empty()) {
         std::cerr << "Error: --pools argument is required for create command" << std::endl;
         return;
     }
     if (fcLayers.empty()) {
         std::cerr << "Error: --fc argument is required for create command" << std::endl;
         return;
     }
     if (outputSizeStr.empty()) {
         std::cerr << "Error: --output argument is required for create command" << std::endl;
         return;
     }

     int inputW = std::stoi(inputWStr);
     int inputH = std::stoi(inputHStr);
     int inputC = std::stoi(inputCStr);
     int outputSize = std::stoi(outputSizeStr);

     std::string hiddenActStr = GetArgValue(argc, argv, "--hidden-act", "relu");
     std::string outputActStr = GetArgValue(argc, argv, "--output-act", "linear");
     std::string lossStr = GetArgValue(argc, argv, "--loss", "mse");
     double lr = std::stod(GetArgValue(argc, argv, "--lr", "0.001"));
     double clip = std::stod(GetArgValue(argc, argv, "--clip", "5.0"));

     std::vector<int> convFilterVec = ParseIntList(convFilters);
     std::vector<int> kernelVec = ParseIntList(kernels);
     std::vector<int> poolVec = ParseIntList(pools);
     std::vector<int> fcVec = ParseIntList(fcLayers);

    TActivationType hiddenAct = ParseActivationType(hiddenActStr);
    TActivationType outputAct = ParseActivationType(outputActStr);
    TLossType lossType = ParseLossType(lossStr);

    std::cout << "Creating CNN model...\n";
    std::cout << "  Input: " << inputW << "x" << inputH << "x" << inputC << "\n";
    
    std::cout << "  Conv filters: ";
    for (size_t i = 0; i < convFilterVec.size(); i++) {
        if (i > 0) std::cout << ",";
        std::cout << convFilterVec[i];
    }
    std::cout << "\n";
    
    std::cout << "  Kernel sizes: ";
    for (size_t i = 0; i < kernelVec.size(); i++) {
        if (i > 0) std::cout << ",";
        std::cout << kernelVec[i];
    }
    std::cout << "\n";
    
    std::cout << "  Pool sizes: ";
    for (size_t i = 0; i < poolVec.size(); i++) {
        if (i > 0) std::cout << ",";
        std::cout << poolVec[i];
    }
    std::cout << "\n";
    
    std::cout << "  FC layers: ";
    for (size_t i = 0; i < fcVec.size(); i++) {
       if (i > 0) std::cout << ",";
       std::cout << fcVec[i];
    }
    std::cout << "\n";
    
    std::cout << "Output size: " << outputSize << "\n";
    std::cout << "  Hidden activation: " << ActivationTypeToStr(hiddenAct) << "\n";
    std::cout << "  Output activation: " << ActivationTypeToStr(outputAct) << "\n";
    std::cout << "  Loss function: " << LossTypeToStr(lossType) << "\n";
    std::cout << std::fixed << std::setprecision(6) << "  Learning rate: " << lr << "\n";
    std::cout << std::fixed << std::setprecision(2) << "  Gradient clip: " << clip << "\n";

    TAdvancedCNN* cnn = new TAdvancedCNN(
        inputW, inputH, inputC,
        convFilterVec, kernelVec, poolVec, fcVec,
        outputSize, hiddenAct, outputAct,
        lossType, lr, clip
    );

    cnn->SaveModelToJSON(saveFile);
    std::cout << "Created CNN model\n";
    std::cout << "Model saved to: " << saveFile << "\n";

    delete cnn;
}

void HandleTrain(int argc, char* argv[]) {
    std::string modelFile = GetArgValue(argc, argv, "--model", "");
    std::string dataFile = GetArgValue(argc, argv, "--data", "");
    std::string epochsStr = GetArgValue(argc, argv, "--epochs", "");
    std::string saveFile = GetArgValue(argc, argv, "--save", "");
    int batchSize = std::stoi(GetArgValue(argc, argv, "--batch-size", "32"));

    // Validate required arguments
    if (modelFile.empty()) {
        std::cerr << "Error: --model argument is required for train command" << std::endl;
        return;
    }
    if (dataFile.empty()) {
        std::cerr << "Error: --data argument is required for train command" << std::endl;
        return;
    }
    if (epochsStr.empty()) {
        std::cerr << "Error: --epochs argument is required for train command" << std::endl;
        return;
    }
    if (saveFile.empty()) {
        std::cerr << "Error: --save argument is required for train command" << std::endl;
        return;
    }

    int epochs = std::stoi(epochsStr);

    std::cout << "Training model...\n";
    std::cout << "  Model: " << modelFile << "\n";
    std::cout << "  Data: " << dataFile << "\n";
    std::cout << "  Epochs: " << epochs << "\n";
    std::cout << "  Batch size: " << batchSize << "\n";
    std::cout << "  Save to: " << saveFile << "\n\n";

    std::cout << "Training not fully implemented in this CLI demo.\n";
    std::cout << "To implement training:\n";
    std::cout << "  1. Load CSV data from " << dataFile << "\n";
    std::cout << "  2. Load model from " << modelFile << "\n";
    std::cout << "  3. Run training loop with TrainBatch() for " << epochs << " epochs\n";
    std::cout << "  4. Save updated model to " << saveFile << "\n";
    std::cout << "\nSee the library API for complete training implementation.\n";
}

void HandlePredict(int argc, char* argv[]) {
    std::string modelFile = GetArgValue(argc, argv, "--model", "");
    std::string dataFile = GetArgValue(argc, argv, "--data", "");
    std::string outputFile = GetArgValue(argc, argv, "--output", "");

    // Validate required arguments
    if (modelFile.empty()) {
        std::cerr << "Error: --model argument is required for predict command" << std::endl;
        return;
    }
    if (dataFile.empty()) {
        std::cerr << "Error: --data argument is required for predict command" << std::endl;
        return;
    }
    if (outputFile.empty()) {
        std::cerr << "Error: --output argument is required for predict command" << std::endl;
        return;
    }

    std::cout << "Making predictions...\n";
    std::cout << "  Model: " << modelFile << "\n";
    std::cout << "  Data: " << dataFile << "\n";
    std::cout << "  Output: " << outputFile << "\n\n";

    std::cout << "Prediction not fully implemented in this CLI demo.\n";
    std::cout << "To implement prediction:\n";
    std::cout << "  1. Load model from " << modelFile << "\n";
    std::cout << "  2. Load input data from CSV file: " << dataFile << "\n";
    std::cout << "  3. Run Predict() on each input\n";
    std::cout << "  4. Save predictions to CSV: " << outputFile << "\n";
    std::cout << "\nSee the library API for complete prediction implementation.\n";
}

// ========== Main Program ==========
int main(int argc, char* argv[]) {
    // Seed random number generator
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    if (argc < 2) {
        PrintHelp();
        return 0;
    }

    // Handle --help flag
    if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        PrintHelp();
        return 0;
    }

    TCommand cmd = ParseCommand(argv[1]);

    try {
        switch (cmd) {
            case TCommand::cmdHelp:
                PrintHelp();
                break;

            case TCommand::cmdInfo: {
                std::string modelFile = GetArgValue(argc, argv, "--model", "");
                if (modelFile.empty()) {
                    std::cerr << "Error: --model argument required for info command\n";
                    return 1;
                }
                PrintModelInfo(modelFile);
                break;
            }

            case TCommand::cmdCreate:
                HandleCreate(argc, argv);
                break;

            case TCommand::cmdTrain:
                HandleTrain(argc, argv);
                break;

            case TCommand::cmdPredict:
                HandlePredict(argc, argv);
                break;

            case TCommand::cmdExportONNX: {
                std::string modelFile = GetArgValue(argc, argv, "--model", "");
                std::string onnxFile = GetArgValue(argc, argv, "--onnx", "");
                if (modelFile.empty()) {
                    std::cerr << "Error: --model argument required for export-onnx command\n";
                    return 1;
                }
                if (onnxFile.empty()) {
                    std::cerr << "Error: --onnx argument required for export-onnx command\n";
                    return 1;
                }
                
                // Create a dummy CNN to load the model
                std::vector<int> dummyConv = {1};
                std::vector<int> dummyKernel = {3};
                std::vector<int> dummyPool = {2};
                std::vector<int> dummyFC = {1};
                TAdvancedCNN cnn(1, 1, 1, dummyConv, dummyKernel, dummyPool, dummyFC, 1,
                                TActivationType::atReLU, TActivationType::atLinear,
                                TLossType::ltCrossEntropy, 0.001, 5.0);
                cnn.LoadModelFromJSON(modelFile);
                cnn.ExportToONNX(onnxFile);
                break;
            }

            case TCommand::cmdImportONNX: {
                std::string onnxFile = GetArgValue(argc, argv, "--onnx", "");
                std::string saveFile = GetArgValue(argc, argv, "--save", "");
                if (onnxFile.empty()) {
                    std::cerr << "Error: --onnx argument required for import-onnx command\n";
                    return 1;
                }
                if (saveFile.empty()) {
                    std::cerr << "Error: --save argument required for import-onnx command\n";
                    return 1;
                }
                
                TAdvancedCNN* cnn = TAdvancedCNN::ImportFromONNX(onnxFile);
                cnn->SaveModelToJSON(saveFile);
                std::cout << "Model imported from ONNX and saved to: " << saveFile << std::endl;
                delete cnn;
                break;
            }

            case TCommand::cmdNone:
            default:
                std::cerr << "Error: Unknown command '" << argv[1] << "'\n";
                std::cerr << "Run 'CNN help' for usage information\n";
                return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
