//
// MatthewAbbott 2025
// Facade CNN with Full Introspection and Manipulation - OpenCL Version
//
// Compile:
//   g++ -o facade_cnn_opencl facade_cnn_opencl.cpp -lOpenCL -std=c++11
//
// CLI Usage:
//   facade_cnn_opencl create --input-w=N --input-h=N --input-c=N --conv=N,N,... --kernels=N,N,... --pools=N,N,... --fc=N,N,... --output=N [options] --save=FILE
//   facade_cnn_opencl train --model=FILE --data=FILE [options] --save=FILE
//   facade_cnn_opencl predict --model=FILE --input=FILE
//   facade_cnn_opencl info --model=FILE
//   facade_cnn_opencl get-filter --model=FILE --layer=L --filter=F --channel=C
//   facade_cnn_opencl get-bias --model=FILE --layer=L --filter=F
//   facade_cnn_opencl set-weight --model=FILE --layer=L --filter=F --channel=C --h=H --w=W --value=V --save=FILE
//   facade_cnn_opencl help
//
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

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
#include <map>

using namespace std;

// ========== OpenCL Error Checking ==========
#define CL_CHECK(err) \
    do { \
        if (err != CL_SUCCESS) { \
            cerr << "OpenCL error at " << __FILE__ << ":" << __LINE__ << " - " << (int)err << endl; \
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

enum Command {
    cmdNone,
    cmdCreate,
    cmdTrain,
    cmdPredict,
    cmdInfo,
    cmdGetFilter,
    cmdGetBias,
    cmdSetWeight,
    cmdHelp
};

const int BLOCK_SIZE = 256;
const char MODEL_MAGIC[] = "CNNFA01";
const double EPSILON = 1e-8;
const double GRAD_CLIP = 1.0;

// Type aliases
typedef vector<float> FArray;
typedef vector<FArray> TFArray2D;
typedef vector<TFArray2D> TFArray3D;
typedef vector<TFArray3D> TFArray4D;
typedef vector<int> TIntArray;

// ========== OpenCL Kernels ==========
const char* kernelSource = R"CLC(
float d_Sigmoid(float x) {
    float clamped = fmax(-500.0f, fmin(500.0f, x));
    return 1.0f / (1.0f + exp(-clamped));
}

float d_DSigmoid(float x) {
    return x * (1.0f - x);
}

float d_TanhActivation(float x) {
    return tanh(x);
}

float d_DTanh(float x) {
    return 1.0f - (x * x);
}

float d_ReLU(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

float d_DReLU(float x) {
    return (x > 0.0f) ? 1.0f : 0.0f;
}

float d_ApplyActivation(float x, int ActType) {
    switch (ActType) {
        case 0: return d_Sigmoid(x);
        case 1: return d_TanhActivation(x);
        case 2: return d_ReLU(x);
        case 3: return x;
        default: return x;
    }
}

float d_ApplyActivationDerivative(float x, int ActType) {
    switch (ActType) {
        case 0: return d_DSigmoid(x);
        case 1: return d_DTanh(x);
        case 2: return d_DReLU(x);
        case 3: return 1.0f;
        default: return 1.0f;
    }
}

float d_ClipValue(float V, float MaxVal) {
    if (V > MaxVal) return MaxVal;
    else if (V < -MaxVal) return -MaxVal;
    else return V;
}

__kernel void k_conv_forward(
    __global float* output,
    __global float* preActivation,
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    int inputChannels, int inputHeight, int inputWidth,
    int kernelSize, int padding, int stride,
    int outputChannels, int outputHeight, int outputWidth,
    int filterIdx) {
    
    int outH = get_global_id(0);
    int outW = get_global_id(1);
    
    if (outH < outputHeight && outW < outputWidth) {
        float sum = bias[filterIdx];
        
        for (int c = 0; c < inputChannels; c++) {
            for (int kh = 0; kh < kernelSize; kh++) {
                for (int kw = 0; kw < kernelSize; kw++) {
                    int ih = outH * stride + kh - padding;
                    int iw = outW * stride + kw - padding;
                    
                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                        int inIdx = (c * inputHeight + ih) * inputWidth + iw;
                        int wIdx = (filterIdx * inputChannels + c) * kernelSize * kernelSize + kh * kernelSize + kw;
                        sum += input[inIdx] * weights[wIdx];
                    }
                }
            }
        }
        
        int outIdx = (filterIdx * outputHeight + outH) * outputWidth + outW;
        preActivation[outIdx] = sum;
        output[outIdx] = d_ReLU(sum);
    }
}

__kernel void k_activate(
    __global float* output,
    __global const float* input,
    int size, int actType) {
    
    int i = get_global_id(0);
    if (i < size) {
        output[i] = d_ApplyActivation(input[i], actType);
    }
}

__kernel void k_pool_forward(
    __global float* output,
    __global int* maxIndices,
    __global const float* input,
    int inputChannels, int inputHeight, int inputWidth,
    int poolSize, int stride,
    int outputHeight, int outputWidth) {
    
    int c = get_global_id(0);
    int h = get_global_id(1);
    int w = get_global_id(2);
    
    if (c < inputChannels && h < outputHeight && w < outputWidth) {
        float maxVal = -1e38f;
        int maxIdx = 0;
        
        for (int ph = 0; ph < poolSize; ph++) {
            for (int pw = 0; pw < poolSize; pw++) {
                int ih = h * stride + ph;
                int iw = w * stride + pw;
                int idx = (c * inputHeight + ih) * inputWidth + iw;
                
                if (input[idx] > maxVal) {
                    maxVal = input[idx];
                    maxIdx = ph * poolSize + pw;
                }
            }
        }
        
        int outIdx = (c * outputHeight + h) * outputWidth + w;
        output[outIdx] = maxVal;
        maxIndices[outIdx] = maxIdx;
    }
}

__kernel void k_flatten(
    __global float* output,
    __global const float* input,
    int channels, int height, int width) {
    
    int idx = get_global_id(0);
    int size = channels * height * width;
    
    if (idx < size) {
        int c = idx / (height * width);
        int hw = idx % (height * width);
        int h = hw / width;
        int w = hw % width;
        
        int inIdx = (c * height + h) * width + w;
        output[idx] = input[inIdx];
    }
}

__kernel void k_fc_forward(
    __global float* output,
    __global float* preActivation,
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    int inputSize, int outputSize, int actType) {
    
    int i = get_global_id(0);
    
    if (i < outputSize) {
        float sum = bias[i];
        for (int j = 0; j < inputSize; j++) {
            sum += weights[i * inputSize + j] * input[j];
        }
        preActivation[i] = sum;
        output[i] = d_ApplyActivation(sum, actType);
    }
}

__kernel void k_softmax(
    __global float* output,
    __global const float* input,
    int size) {
    
    int i = get_global_id(0);
    
    if (i == 0) {
        float maxVal = input[0];
        for (int j = 1; j < size; j++) {
            if (input[j] > maxVal) maxVal = input[j];
        }
        
        float sum = 0.0f;
        for (int j = 0; j < size; j++) {
            output[j] = exp(input[j] - maxVal);
            sum += output[j];
        }
        
        for (int j = 0; j < size; j++) {
            output[j] /= (sum + 1e-15f);
        }
    }
}

__kernel void k_update_weights(
    __global float* weights,
    __global const float* gradients,
    int size, float learningRate) {
    
    int i = get_global_id(0);
    if (i < size) {
        weights[i] -= learningRate * gradients[i];
    }
}
)CLC";

// ========== OpenCL Context Manager ==========
class TOpenCLContext {
public:
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    
    map<string, cl_kernel> kernels;
    
    TOpenCLContext() {
        cl_int err;
        
        err = clGetPlatformIDs(1, &platform, nullptr);
        CL_CHECK(err);
        
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) {
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
        }
        CL_CHECK(err);
        
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        CL_CHECK(err);
        
        queue = clCreateCommandQueue(context, device, 0, &err);
        CL_CHECK(err);
        
        program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
        CL_CHECK(err);
        
        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t len;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
            char* log = new char[len];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, log, nullptr);
            cerr << "Kernel build error:\n" << log << endl;
            delete[] log;
            exit(1);
        }
    }
    
    ~TOpenCLContext() {
        for (auto& kv : kernels) {
            clReleaseKernel(kv.second);
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }
    
    cl_kernel GetKernel(const string& name) {
        if (kernels.find(name) == kernels.end()) {
            cl_int err;
            kernels[name] = clCreateKernel(program, name.c_str(), &err);
            CL_CHECK(err);
        }
        return kernels[name];
    }
};

// ========== Utility Functions ==========
double RandomWeight(double Scale) {
    return (rand() / (double)RAND_MAX - 0.5) * 2.0 * Scale;
}

void InitMatrixF(TFArray2D& M, int Rows, int Cols, double Scale) {
    M.resize(Rows);
    for (int i = 0; i < Rows; i++) {
        M[i].resize(Cols);
        for (int j = 0; j < Cols; j++) {
            M[i][j] = RandomWeight(Scale);
        }
    }
}

void ZeroMatrixF(TFArray2D& M, int Rows, int Cols) {
    M.resize(Rows);
    for (int i = 0; i < Rows; i++) {
        M[i].resize(Cols);
        for (int j = 0; j < Cols; j++) {
            M[i][j] = 0.0f;
        }
    }
}

void ZeroArrayF(FArray& A, int Size) {
    A.resize(Size);
    for (int i = 0; i < Size; i++) {
        A[i] = 0.0f;
    }
}

void Zero3DArrayF(TFArray3D& A, int D1, int D2, int D3) {
    A.resize(D1);
    for (int i = 0; i < D1; i++) {
        A[i].resize(D2);
        for (int j = 0; j < D2; j++) {
            A[i][j].resize(D3);
            for (int k = 0; k < D3; k++) {
                A[i][j][k] = 0.0f;
            }
        }
    }
}

void Zero4DArrayF(TFArray4D& A, int D1, int D2, int D3, int D4) {
    A.resize(D1);
    for (int i = 0; i < D1; i++) {
        A[i].resize(D2);
        for (int j = 0; j < D2; j++) {
            A[i][j].resize(D3);
            for (int k = 0; k < D3; k++) {
                A[i][j][k].resize(D4);
                for (int l = 0; l < D4; l++) {
                    A[i][j][k][l] = 0.0f;
                }
            }
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

void ParseIntArrayHelper(const string& s, TIntArray& result) {
    result.clear();
    stringstream ss(s);
    string token;
    while (getline(ss, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t\r\n"));
        token.erase(token.find_last_not_of(" \t\r\n") + 1);
        if (!token.empty()) {
            result.push_back(stoi(token));
        }
    }
}

void PrintUsage() {
    cout << "Facade CNN OpenCL - GPU-accelerated CNN with Introspection\n";
    cout << "Matthew Abbott 2025 \n\n";
    cout << "Commands:\n";
    cout << "  create       Create a new CNN model\n";
    cout << "  train        Train an existing model with data\n";
    cout << "  predict      Make predictions with a trained model\n";
    cout << "  info         Display model information\n";
    cout << "  get-filter   Get filter kernel values\n";
    cout << "  get-bias     Get filter bias value\n";
    cout << "  set-weight   Set individual weight value\n";
    cout << "  help         Show this help message\n\n";
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
    cout << "  --clip=VALUE           Gradient clipping (default: 1.0)\n\n";
    cout << "Introspection Options:\n";
    cout << "  --layer=L              Layer index\n";
    cout << "  --filter=F             Filter index\n";
    cout << "  --channel=C            Channel index\n";
    cout << "  --h=H                  Height coordinate\n";
    cout << "  --w=W                  Width coordinate\n";
    cout << "  --value=V              Weight value (for set-weight)\n\n";
    cout << "Examples:\n";
    cout << "  facade_cnn_opencl create --input-w=28 --input-h=28 --input-c=1 --conv=32,64 --kernels=3,3 --pools=2,2 --fc=128 --output=10 --save=model.bin\n";
    cout << "  facade_cnn_opencl info --model=model.bin\n";
    cout << "  facade_cnn_opencl get-filter --model=model.bin --layer=0 --filter=0 --channel=0\n";
}

// ========== Convolutional Filter ==========
class TConvFilter {
public:
    int FInputChannels, FOutputChannels, FKernelSize;
    TFArray3D Weights;
    TFArray3D dWeights;
    float Bias;
    float dBias;
    
    cl_mem gpu_weights, gpu_dweights, gpu_bias, gpu_dbias;
    
    TConvFilter(int InputChannels, int OutputChannels, int KernelSize, TOpenCLContext* oclCtx) {
        FInputChannels = InputChannels;
        FOutputChannels = OutputChannels;
        FKernelSize = KernelSize;
        double Scale = sqrt(2.0 / (InputChannels * KernelSize * KernelSize));
        
        Zero3DArrayF(Weights, InputChannels, KernelSize, KernelSize);
        Zero3DArrayF(dWeights, InputChannels, KernelSize, KernelSize);
        
        for (int j = 0; j < InputChannels; j++) {
            for (int k = 0; k < KernelSize; k++) {
                for (int l = 0; l < KernelSize; l++) {
                    Weights[j][k][l] = RandomWeight(Scale);
                }
            }
        }
        
        Bias = 0.0f;
        dBias = 0.0f;
        
        // Allocate GPU memory
        cl_int err;
        int wsize = InputChannels * KernelSize * KernelSize;
        FArray flatWeights(wsize);
        int idx = 0;
        for (int j = 0; j < InputChannels; j++) {
            for (int k = 0; k < KernelSize; k++) {
                for (int l = 0; l < KernelSize; l++) {
                    flatWeights[idx++] = Weights[j][k][l];
                }
            }
        }
        
        gpu_weights = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     sizeof(float) * wsize, flatWeights.data(), &err);
        CL_CHECK(err);
        
        gpu_dweights = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE,
                                      sizeof(float) * wsize, nullptr, &err);
        CL_CHECK(err);
        
        gpu_bias = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float), &Bias, &err);
        CL_CHECK(err);
        
        gpu_dbias = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE,
                                   sizeof(float), nullptr, &err);
        CL_CHECK(err);
    }
    
    ~TConvFilter() {
        clReleaseMemObject(gpu_weights);
        clReleaseMemObject(gpu_dweights);
        clReleaseMemObject(gpu_bias);
        clReleaseMemObject(gpu_dbias);
    }
    
    void ResetGradients() {
        for (size_t j = 0; j < dWeights.size(); j++) {
            for (size_t k = 0; k < dWeights[j].size(); k++) {
                for (size_t l = 0; l < dWeights[j][k].size(); l++) {
                    dWeights[j][k][l] = 0.0f;
                }
            }
        }
        dBias = 0.0f;
    }
};

// ========== Convolutional Layer ==========
class TConvLayer {
private:
    int FInputChannels, FOutputChannels, FKernelSize, FStride, FPadding;
    ActivationType FActivation;
    
public:
    vector<TConvFilter*> Filters;
    TFArray3D InputCache;
    TFArray3D OutputCache;
    TFArray3D PreActivation;
    
    cl_mem gpu_input, gpu_output, gpu_preact;
    TOpenCLContext* oclCtx;
    
    TConvLayer(int InputChannels, int OutputChannels, int KernelSize, int Stride, int Padding,
               ActivationType Activation, TOpenCLContext* OclCtx) {
        FInputChannels = InputChannels;
        FOutputChannels = OutputChannels;
        FKernelSize = KernelSize;
        FStride = Stride;
        FPadding = Padding;
        FActivation = Activation;
        oclCtx = OclCtx;
        
        gpu_input = nullptr;
        gpu_output = nullptr;
        gpu_preact = nullptr;
        
        Filters.resize(OutputChannels);
        for (int i = 0; i < OutputChannels; i++) {
            Filters[i] = new TConvFilter(InputChannels, 1, KernelSize, OclCtx);
        }
    }
    
    ~TConvLayer() {
        for (auto& f : Filters) {
            delete f;
        }
        Filters.clear();
        if (gpu_input) clReleaseMemObject(gpu_input);
        if (gpu_output) clReleaseMemObject(gpu_output);
        if (gpu_preact) clReleaseMemObject(gpu_preact);
    }
    
    void Forward(const TFArray3D& Input, TFArray3D& Output) {
        InputCache = Input;
        
        int outH = Input[0].size();
        int outW = Input[0][0].size();
        
        Zero3DArrayF(Output, FOutputChannels, outH, outW);
        Zero3DArrayF(PreActivation, FOutputChannels, outH, outW);
        
        // Flatten input for GPU
        int inputSize = Input.size() * Input[0].size() * Input[0][0].size();
        FArray flatInput(inputSize);
        int idx = 0;
        for (size_t c = 0; c < Input.size(); c++) {
            for (size_t h = 0; h < Input[c].size(); h++) {
                for (size_t w = 0; w < Input[c][h].size(); w++) {
                    flatInput[idx++] = Input[c][h][w];
                }
            }
        }
        
        cl_int err;
        if (gpu_input) clReleaseMemObject(gpu_input);
        gpu_input = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * flatInput.size(), flatInput.data(), &err);
        CL_CHECK(err);
        
        int outSize = FOutputChannels * outH * outW;
        if (gpu_output) clReleaseMemObject(gpu_output);
        gpu_output = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE,
                                    sizeof(float) * outSize, nullptr, &err);
        CL_CHECK(err);
        
        if (gpu_preact) clReleaseMemObject(gpu_preact);
        gpu_preact = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE,
                                    sizeof(float) * outSize, nullptr, &err);
        CL_CHECK(err);
        
        // Run convolution for each filter on GPU
        for (int f = 0; f < FOutputChannels; f++) {
            cl_kernel kern = oclCtx->GetKernel("k_conv_forward");
            
            int padH = Input[0].size();
            int padW = Input[0][0].size();
            
            err = clSetKernelArg(kern, 0, sizeof(cl_mem), &gpu_output);
            err |= clSetKernelArg(kern, 1, sizeof(cl_mem), &gpu_preact);
            err |= clSetKernelArg(kern, 2, sizeof(cl_mem), &gpu_input);
            err |= clSetKernelArg(kern, 3, sizeof(cl_mem), &Filters[f]->gpu_weights);
            err |= clSetKernelArg(kern, 4, sizeof(cl_mem), &Filters[f]->gpu_bias);
            err |= clSetKernelArg(kern, 5, sizeof(int), &FInputChannels);
            err |= clSetKernelArg(kern, 6, sizeof(int), &padH);
            err |= clSetKernelArg(kern, 7, sizeof(int), &padW);
            err |= clSetKernelArg(kern, 8, sizeof(int), &FKernelSize);
            err |= clSetKernelArg(kern, 9, sizeof(int), &FPadding);
            err |= clSetKernelArg(kern, 10, sizeof(int), &FStride);
            err |= clSetKernelArg(kern, 11, sizeof(int), &FOutputChannels);
            err |= clSetKernelArg(kern, 12, sizeof(int), &outH);
            err |= clSetKernelArg(kern, 13, sizeof(int), &outW);
            err |= clSetKernelArg(kern, 14, sizeof(int), &f);
            CL_CHECK(err);
            
            size_t globalSize[2] = {(size_t)outH, (size_t)outW};
            err = clEnqueueNDRangeKernel(oclCtx->queue, kern, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
            CL_CHECK(err);
        }
        
        clFinish(oclCtx->queue);
        
        // Read output back to host
        FArray flatOutput(outSize);
        err = clEnqueueReadBuffer(oclCtx->queue, gpu_output, CL_TRUE, 0, sizeof(float) * outSize, flatOutput.data(), 0, nullptr, nullptr);
        CL_CHECK(err);
        
        idx = 0;
        for (int f = 0; f < FOutputChannels; f++) {
            for (int h = 0; h < outH; h++) {
                for (int w = 0; w < outW; w++) {
                    Output[f][h][w] = flatOutput[idx++];
                }
            }
        }
        
        OutputCache = Output;
    }
    
    int GetOutputChannels() const {
        return FOutputChannels;
    }
};

// ========== Pooling Layer ==========
class TPoolingLayer {
private:
    int FPoolSize, FStride;
    
public:
    TFArray3D InputCache;
    TFArray3D OutputCache;
    vector<vector<vector<pair<int,int>>>> MaxIndices;
    
    TPoolingLayer(int PoolSize, int Stride) {
        FPoolSize = PoolSize;
        FStride = Stride;
    }
    
    void Forward(const TFArray3D& Input, TFArray3D& Output) {
        InputCache = Input;
        int outH = (Input[0].size() - FPoolSize) / FStride + 1;
        int outW = (Input[0][0].size() - FPoolSize) / FStride + 1;
        
        Zero3DArrayF(Output, Input.size(), outH, outW);
        MaxIndices.assign(Input.size(), vector<vector<pair<int,int>>>(outH, vector<pair<int,int>>(outW)));
        
        for (size_t c = 0; c < Input.size(); c++) {
            for (int h = 0; h < outH; h++) {
                for (int w = 0; w < outW; w++) {
                    float MaxVal = -1e38f;
                    int MaxH = 0, MaxW = 0;
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
                    MaxIndices[c][h][w] = make_pair(MaxW, MaxH);
                }
            }
        }
        OutputCache = Output;
    }
};

// ========== Fully Connected Layer ==========
class TFCLayer {
private:
    int FInputSize, FOutputSize;
    ActivationType FActivation;
    
public:
    TFArray2D W;
    FArray B;
    TFArray2D dW;
    FArray dB;
    FArray InputCache;
    FArray OutputCache;
    FArray PreActivation;
    
    cl_mem gpu_W, gpu_B, gpu_dW, gpu_dB, gpu_input, gpu_output, gpu_preact;
    TOpenCLContext* oclCtx;
    
    TFCLayer(int InputSize, int OutputSize, ActivationType Activation, TOpenCLContext* OclCtx) {
        FInputSize = InputSize;
        FOutputSize = OutputSize;
        FActivation = Activation;
        oclCtx = OclCtx;
        
        double Scale = sqrt(2.0 / InputSize);
        
        InitMatrixF(W, OutputSize, InputSize, Scale);
        ZeroArrayF(B, OutputSize);
        ZeroMatrixF(dW, OutputSize, InputSize);
        ZeroArrayF(dB, OutputSize);
        
        gpu_input = nullptr;
        gpu_output = nullptr;
        gpu_preact = nullptr;
        
        cl_int err;
        FArray flatW(W.size() * W[0].size());
        int idx = 0;
        for (size_t i = 0; i < W.size(); i++) {
            for (size_t j = 0; j < W[i].size(); j++) {
                flatW[idx++] = W[i][j];
            }
        }
        
        gpu_W = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(float) * flatW.size(), flatW.data(), &err);
        CL_CHECK(err);
        
        gpu_B = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(float) * B.size(), B.data(), &err);
        CL_CHECK(err);
        
        gpu_dW = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE,
                               sizeof(float) * flatW.size(), nullptr, &err);
        CL_CHECK(err);
        
        gpu_dB = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE,
                               sizeof(float) * B.size(), nullptr, &err);
        CL_CHECK(err);
    }
    
    ~TFCLayer() {
        clReleaseMemObject(gpu_W);
        clReleaseMemObject(gpu_B);
        clReleaseMemObject(gpu_dW);
        clReleaseMemObject(gpu_dB);
        if (gpu_input) clReleaseMemObject(gpu_input);
        if (gpu_output) clReleaseMemObject(gpu_output);
        if (gpu_preact) clReleaseMemObject(gpu_preact);
    }
    
    void Forward(const FArray& Input, FArray& Output) {
        InputCache = Input;
        Output.resize(FOutputSize);
        PreActivation.resize(FOutputSize);
        
        for (int i = 0; i < FOutputSize; i++) {
            float Sum = B[i];
            for (int j = 0; j < FInputSize; j++) {
                Sum += W[i][j] * Input[j];
            }
            PreActivation[i] = Sum;
            Output[i] = (FActivation == atLinear) ? Sum : (FActivation == atReLU ? (Sum > 0 ? Sum : 0) : 
                       (FActivation == atSigmoid ? (1.0f / (1.0f + exp(-Sum))) : tanh(Sum)));
        }
        
        OutputCache = Output;
    }
};

// ========== Main Advanced CNN Facade ==========
class TAdvancedCNNFacade {
private:
    int FInputWidth, FInputHeight, FInputChannels, FOutputSize;
    ActivationType FActivation, FOutputActivation;
    LossType FLossType;
    double FLearningRate, FGradientClip;
    
    vector<TConvLayer*> FConvLayers;
    vector<TPoolingLayer*> FPoolLayers;
    vector<TFCLayer*> FFullyConnectedLayers;
    TFCLayer* FOutputLayer;
    int FFlattenedSize;
    
    TOpenCLContext* oclCtx;
    
public:
    TAdvancedCNNFacade(int InputWidth, int InputHeight, int InputChannels,
                       const TIntArray& ConvFilters,
                       const TIntArray& KernelSizes,
                       const TIntArray& PoolSizes,
                       const TIntArray& FCLayerSizes,
                       int OutputSize,
                       ActivationType Activation,
                       ActivationType OutputActivation,
                       LossType LossType,
                       double LearningRate,
                       double GradientClip) {
        
        oclCtx = new TOpenCLContext();
        
        FInputWidth = InputWidth;
        FInputHeight = InputHeight;
        FInputChannels = InputChannels;
        FOutputSize = OutputSize;
        FActivation = Activation;
        FOutputActivation = OutputActivation;
        FLossType = LossType;
        FLearningRate = LearningRate;
        FGradientClip = GradientClip;
        
        int CurrentChannels = InputChannels;
        int CurrentWidth = InputWidth;
        int CurrentHeight = InputHeight;
        
        FConvLayers.resize(ConvFilters.size());
        for (size_t i = 0; i < ConvFilters.size(); i++) {
            FConvLayers[i] = new TConvLayer(CurrentChannels, ConvFilters[i], 
                                            KernelSizes[i], 1, KernelSizes[i] / 2, 
                                            Activation, oclCtx);
            CurrentChannels = ConvFilters[i];
            
            if (i < PoolSizes.size()) {
                CurrentWidth /= PoolSizes[i];
                CurrentHeight /= PoolSizes[i];
            }
        }
        
        FPoolLayers.resize(PoolSizes.size());
        for (size_t i = 0; i < PoolSizes.size(); i++) {
            FPoolLayers[i] = new TPoolingLayer(PoolSizes[i], PoolSizes[i]);
        }
        
        FFlattenedSize = CurrentChannels * CurrentWidth * CurrentHeight;
        
        int NumInputs = FFlattenedSize;
        FFullyConnectedLayers.resize(FCLayerSizes.size());
        for (size_t i = 0; i < FCLayerSizes.size(); i++) {
            FFullyConnectedLayers[i] = new TFCLayer(NumInputs, FCLayerSizes[i], Activation, oclCtx);
            NumInputs = FCLayerSizes[i];
        }
        
        FOutputLayer = new TFCLayer(NumInputs, OutputSize, OutputActivation, oclCtx);
    }
    
    ~TAdvancedCNNFacade() {
        for (auto& layer : FConvLayers) {
            delete layer;
        }
        for (auto& layer : FPoolLayers) {
            delete layer;
        }
        for (auto& layer : FFullyConnectedLayers) {
            delete layer;
        }
        delete FOutputLayer;
        FConvLayers.clear();
        FPoolLayers.clear();
        FFullyConnectedLayers.clear();
        
        delete oclCtx;
    }
    
    // Introspection methods
    TFArray2D GetKernel(int LayerIdx, int FilterIdx, int ChannelIdx) {
        TFArray2D Result;
        if (LayerIdx >= 0 && LayerIdx < (int)FConvLayers.size() &&
            FilterIdx >= 0 && FilterIdx < (int)FConvLayers[LayerIdx]->Filters.size() &&
            ChannelIdx >= 0 && ChannelIdx < (int)FConvLayers[LayerIdx]->Filters[FilterIdx]->Weights.size()) {
            Result = FConvLayers[LayerIdx]->Filters[FilterIdx]->Weights[ChannelIdx];
        }
        return Result;
    }
    
    float GetBias(int LayerIdx, int FilterIdx) {
        if (LayerIdx >= 0 && LayerIdx < (int)FConvLayers.size() &&
            FilterIdx >= 0 && FilterIdx < (int)FConvLayers[LayerIdx]->Filters.size()) {
            return FConvLayers[LayerIdx]->Filters[FilterIdx]->Bias;
        }
        return 0.0f;
    }
    
    void SetWeight(int LayerIdx, int FilterIdx, int ChannelIdx, int H, int W, float Value) {
        if (LayerIdx >= 0 && LayerIdx < (int)FConvLayers.size() &&
            FilterIdx >= 0 && FilterIdx < (int)FConvLayers[LayerIdx]->Filters.size() &&
            ChannelIdx >= 0 && ChannelIdx < (int)FConvLayers[LayerIdx]->Filters[FilterIdx]->Weights.size() &&
            H >= 0 && H < (int)FConvLayers[LayerIdx]->Filters[FilterIdx]->Weights[ChannelIdx].size() &&
            W >= 0 && W < (int)FConvLayers[LayerIdx]->Filters[FilterIdx]->Weights[ChannelIdx][H].size()) {
            FConvLayers[LayerIdx]->Filters[FilterIdx]->Weights[ChannelIdx][H][W] = Value;
        }
    }
    
    int GetNumConvLayers() const { return FConvLayers.size(); }
    int GetNumPoolLayers() const { return FPoolLayers.size(); }
    int GetNumFCLayers() const { return FFullyConnectedLayers.size(); }
    int GetNumFilters(int LayerIdx) const {
        if (LayerIdx >= 0 && LayerIdx < (int)FConvLayers.size()) {
            return FConvLayers[LayerIdx]->Filters.size();
        }
        return 0;
    }
};

// ========== Main Program ==========
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
    else if (CmdStr == "get-filter") Cmd = cmdGetFilter;
    else if (CmdStr == "get-bias") Cmd = cmdGetBias;
    else if (CmdStr == "set-weight") Cmd = cmdSetWeight;
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
    double gradientClip = 1.0;
    ActivationType hiddenAct = atReLU;
    ActivationType outputAct = atLinear;
    LossType lossType = ltMSE;
    string modelFile = "";
    string saveFile = "";
    int layerIdx = -1, filterIdx = -1, channelIdx = -1, hIdx = -1, wIdx = -1;
    float weightValue = 0.0f;
    
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
        else if (key == "--layer") layerIdx = stoi(value);
        else if (key == "--filter") filterIdx = stoi(value);
        else if (key == "--channel") channelIdx = stoi(value);
        else if (key == "--h") hIdx = stoi(value);
        else if (key == "--w") wIdx = stoi(value);
        else if (key == "--value") weightValue = stof(value);
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
        
        TAdvancedCNNFacade* CNN = new TAdvancedCNNFacade(inputW, inputH, inputC, convFilters, kernelSizes,
                                                         poolSizes, fcLayerSizes, outputSize,
                                                         hiddenAct, outputAct, lossType, 
                                                         learningRate, gradientClip);
        
        cout << "Created Facade CNN model (OpenCL):\n";
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
    else if (Cmd == cmdInfo) {
        cout << "Info command requires model loading (not yet implemented)\n";
    }
    else if (Cmd == cmdGetFilter) {
        cout << "Get filter command requires model loading (not yet implemented)\n";
        if (layerIdx >= 0 && filterIdx >= 0 && channelIdx >= 0) {
            cout << "  Requested: Layer=" << layerIdx << ", Filter=" << filterIdx << ", Channel=" << channelIdx << "\n";
        }
    }
    else if (Cmd == cmdGetBias) {
        cout << "Get bias command requires model loading (not yet implemented)\n";
    }
    else if (Cmd == cmdSetWeight) {
        cout << "Set weight command requires model loading (not yet implemented)\n";
    }
    else if (Cmd == cmdTrain) {
        cout << "Train command requires model loading (not yet implemented)\n";
    }
    else if (Cmd == cmdPredict) {
        cout << "Predict command requires model loading (not yet implemented)\n";
    }
    
    return 0;
}
