//
// MatthewAbbott 2025
// Facade CNN with Full Introspection and Manipulation - CUDA Version
//
// Compile:
//   nvcc -O2 -o facade_cnn facade_cnn.cu -std=c++11
//
// CLI Usage:
//   facade_cnn create --input-w=N --input-h=N --input-c=N --conv=N,N,... --kernels=N,N,... --pools=N,N,... --fc=N,N,... --output=N [options] --save=FILE
//   facade_cnn train --model=FILE --data=FILE [options] --save=FILE
//   facade_cnn predict --model=FILE --input=FILE
//   facade_cnn info --model=FILE
//   facade_cnn get-filter --model=FILE --layer=L --filter=F --channel=C
//   facade_cnn get-bias --model=FILE --layer=L --filter=F
//   facade_cnn set-weight --model=FILE --layer=L --filter=F --channel=C --h=H --w=W --value=V --save=FILE
//   facade_cnn help
//

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
    cout << "Facade CNN CUDA - GPU-accelerated CNN with Introspection\n";
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
    cout << "Facade Introspection Commands (run with predict/train):\n";
    cout << "  --get-feature-map=L,F         Get feature map for layer L, filter F\n";
    cout << "  --get-preactivation=L,F       Get pre-activation for layer L, filter F\n";
    cout << "  --get-kernel=L,F              Get kernel weights for layer L, filter F\n";
    cout << "  --get-bias=L,F                Get bias for layer L, filter F\n";
    cout << "  --get-filter-gradient=L,F     Get weight gradients for layer L, filter F\n";
    cout << "  --get-bias-gradient=L,F       Get bias gradient for layer L, filter F\n";
    cout << "  --get-pooling-indices=L,F     Get max pooling indices for layer L, filter F\n";
    cout << "  --get-flattened               Get flattened feature vector\n";
    cout << "  --get-logits                  Get raw logits from output layer\n";
    cout << "  --get-softmax                 Get softmax probabilities\n";
    cout << "  --get-layer-stats=L           Get statistics for layer L\n";
    cout << "  --get-activation-hist=L       Get activation histogram for layer L\n";
    cout << "  --get-weight-hist=L           Get weight histogram for layer L\n";
    cout << "  --get-receptive-field=L       Get receptive field size at layer L\n";
    cout << "  --get-fc-weights=L            Get fully connected weights for layer L\n";
    cout << "  --get-fc-bias=L               Get fully connected bias for layer L\n";
    cout << "  --get-dropout-mask=L          Get dropout mask for layer L\n";
    cout << "  --add-filter=L,N              Add N new filters to layer L\n";
    cout << "  --get-num-filters=L           Get number of filters in layer L\n";
    cout << "  --set-bias=L,F,V              Set bias for layer L, filter F to value V\n";
    cout << "  --set-fc-bias=L,N,V           Set FC bias for layer L, neuron N to value V\n\n";
    cout << "Examples:\n";
    cout << "  facade_cnn create --input-w=28 --input-h=28 --input-c=1 --conv=32,64 --kernels=3,3 --pools=2,2 --fc=128 --output=10 --save=model.bin\n";
    cout << "  facade_cnn info --model=model.bin\n";
    cout << "  facade_cnn get-filter --model=model.bin --layer=0 --filter=0 --channel=0\n";
}

// ========== Convolutional Filter ==========
class TConvFilter {
public:
    int FInputChannels, FOutputChannels, FKernelSize;
    TFArray3D Weights;
    TFArray3D dWeights;
    float Bias;
    float dBias;
    
    TConvFilter(int InputChannels, int OutputChannels, int KernelSize) {
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
    }
    
    ~TConvFilter() {
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
    
    TConvLayer(int InputChannels, int OutputChannels, int KernelSize, int Stride, int Padding,
               ActivationType Activation) {
        FInputChannels = InputChannels;
        FOutputChannels = OutputChannels;
        FKernelSize = KernelSize;
        FStride = Stride;
        FPadding = Padding;
        FActivation = Activation;
        
        Filters.resize(OutputChannels);
        for (int i = 0; i < OutputChannels; i++) {
            Filters[i] = new TConvFilter(InputChannels, 1, KernelSize);
        }
    }
    
    ~TConvLayer() {
        for (auto& f : Filters) {
            delete f;
        }
        Filters.clear();
    }
    
    void Forward(const TFArray3D& Input, TFArray3D& Output) {
        InputCache = Input;
        
        int outH = Input[0].size();
        int outW = Input[0][0].size();
        
        Zero3DArrayF(Output, FOutputChannels, outH, outW);
        Zero3DArrayF(PreActivation, FOutputChannels, outH, outW);
        
        for (int f = 0; f < FOutputChannels; f++) {
            for (int h = 0; h < outH; h++) {
                for (int w = 0; w < outW; w++) {
                    float sum = Filters[f]->Bias;
                    
                    for (int c = 0; c < FInputChannels; c++) {
                        for (int kh = 0; kh < FKernelSize; kh++) {
                            for (int kw = 0; kw < FKernelSize; kw++) {
                                int ih = h * FStride + kh - FPadding;
                                int iw = w * FStride + kw - FPadding;
                                
                                if (ih >= 0 && ih < (int)Input[c].size() && iw >= 0 && iw < (int)Input[c][h].size()) {
                                    sum += Input[c][ih][iw] * Filters[f]->Weights[c][kh][kw];
                                }
                            }
                        }
                    }
                    
                    PreActivation[f][h][w] = sum;
                    Output[f][h][w] = (sum > 0) ? sum : 0;  // ReLU
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
    
    TFCLayer(int InputSize, int OutputSize, ActivationType Activation) {
        FInputSize = InputSize;
        FOutputSize = OutputSize;
        FActivation = Activation;
        
        double Scale = sqrt(2.0 / InputSize);
        
        InitMatrixF(W, OutputSize, InputSize, Scale);
        ZeroArrayF(B, OutputSize);
        ZeroMatrixF(dW, OutputSize, InputSize);
        ZeroArrayF(dB, OutputSize);
    }
    
    ~TFCLayer() {
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
                                            Activation);
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
            FFullyConnectedLayers[i] = new TFCLayer(NumInputs, FCLayerSizes[i], Activation);
            NumInputs = FCLayerSizes[i];
        }
        
        FOutputLayer = new TFCLayer(NumInputs, OutputSize, OutputActivation);
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
    map<string, string> facadeCommands;
    
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
        // Facade introspection commands
        else if (key == "--get-feature-map") facadeCommands["get-feature-map"] = value;
        else if (key == "--get-preactivation") facadeCommands["get-preactivation"] = value;
        else if (key == "--get-kernel") facadeCommands["get-kernel"] = value;
        else if (key == "--get-bias") facadeCommands["get-bias"] = value;
        else if (key == "--get-filter-gradient") facadeCommands["get-filter-gradient"] = value;
        else if (key == "--get-bias-gradient") facadeCommands["get-bias-gradient"] = value;
        else if (key == "--get-pooling-indices") facadeCommands["get-pooling-indices"] = value;
        else if (key == "--get-flattened") facadeCommands["get-flattened"] = "1";
        else if (key == "--get-logits") facadeCommands["get-logits"] = "1";
        else if (key == "--get-softmax") facadeCommands["get-softmax"] = "1";
        else if (key == "--get-layer-stats") facadeCommands["get-layer-stats"] = value;
        else if (key == "--get-activation-hist") facadeCommands["get-activation-hist"] = value;
        else if (key == "--get-weight-hist") facadeCommands["get-weight-hist"] = value;
        else if (key == "--get-receptive-field") facadeCommands["get-receptive-field"] = value;
        else if (key == "--get-fc-weights") facadeCommands["get-fc-weights"] = value;
        else if (key == "--get-fc-bias") facadeCommands["get-fc-bias"] = value;
        else if (key == "--get-dropout-mask") facadeCommands["get-dropout-mask"] = value;
        else if (key == "--add-filter") facadeCommands["add-filter"] = value;
        else if (key == "--get-num-filters") facadeCommands["get-num-filters"] = value;
        else if (key == "--set-bias") facadeCommands["set-bias"] = value;
        else if (key == "--set-fc-bias") facadeCommands["set-fc-bias"] = value;
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
        
        cout << "Created Facade CNN model (CUDA):\n";
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
