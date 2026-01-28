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

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::ffi::c_void;
use std::fs::File;
use std::io::BufWriter;
use std::sync::Arc;

const BLOCK_SIZE: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ActivationType {
    Sigmoid,
    Tanh,
    ReLU,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum LossType {
    MSE,
    CrossEntropy,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchNormParams {
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
    pub running_mean: Vec<f64>,
    pub running_var: Vec<f64>,
    pub epsilon: f64,
    pub momentum: f64,
}

impl BatchNormParams {
    pub fn new() -> Self {
        Self {
            gamma: Vec::new(),
            beta: Vec::new(),
            running_mean: Vec::new(),
            running_var: Vec::new(),
            epsilon: 1e-5,
            momentum: 0.1,
        }
    }

    pub fn initialize(&mut self, size: usize) {
        self.gamma = vec![1.0; size];
        self.beta = vec![0.0; size];
        self.running_mean = vec![0.0; size];
        self.running_var = vec![1.0; size];
    }
}

impl Default for BatchNormParams {
    fn default() -> Self {
        Self::new()
    }
}

const CUDA_KERNEL_SRC: &str = r#"
extern "C" {

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
    return d_Clamp(x, -1.0, 1.0);
}

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

__global__ void conv_forward_kernel(double* output, double* pre_activation,
                                    const double* input, const double* weights,
                                    const double* biases,
                                    int input_channels, int kernel_size,
                                    int input_h, int input_w, int output_h, int output_w,
                                    int stride, int padding, int num_filters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_filters * output_h * output_w;
    if (idx >= total) return;

    int f = idx / (output_h * output_w);
    int rem = idx % (output_h * output_w);
    int oh = rem / output_w;
    int ow = rem % output_w;

    double sum = biases[f];

    for (int c = 0; c < input_channels; c++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = oh * stride + kh;
                int iw = ow * stride + kw;
                int padded_h = input_h + 2 * padding;
                int padded_w = input_w + 2 * padding;
                int input_idx = c * padded_h * padded_w + ih * padded_w + iw;
                int weight_idx = f * input_channels * kernel_size * kernel_size +
                               c * kernel_size * kernel_size + kh * kernel_size + kw;
                sum += input[input_idx] * weights[weight_idx];
            }
        }
    }

    if (!isfinite(sum)) sum = 0;

    int out_idx = f * output_h * output_w + oh * output_w + ow;
    pre_activation[out_idx] = sum;
    output[out_idx] = d_ReLU(sum);
}

__global__ void pool_forward_kernel(double* output, int* max_indices_y, int* max_indices_x,
                                    const double* input,
                                    int channels, int input_h, int input_w,
                                    int output_h, int output_w, int pool_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * output_h * output_w;
    if (idx >= total) return;

    int c = idx / (output_h * output_w);
    int rem = idx % (output_h * output_w);
    int oh = rem / output_w;
    int ow = rem % output_w;

    double max_val = -1e308;
    int max_ph = 0, max_pw = 0;

    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int ih = oh * pool_size + ph;
            int iw = ow * pool_size + pw;
            int input_idx = c * input_h * input_w + ih * input_w + iw;
            double val = input[input_idx];
            if (val > max_val) {
                max_val = val;
                max_ph = ph;
                max_pw = pw;
            }
        }
    }

    int out_idx = c * output_h * output_w + oh * output_w + ow;
    output[out_idx] = max_val;
    max_indices_y[out_idx] = max_ph;
    max_indices_x[out_idx] = max_pw;
}

__global__ void fc_forward_kernel(double* output, double* pre_activation,
                                  const double* input, const double* weights,
                                  const double* biases, const double* dropout_mask,
                                  int num_neurons, int num_inputs, int apply_relu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_neurons) return;

    double sum = biases[i];
    for (int j = 0; j < num_inputs; j++) {
        sum += input[j] * weights[i * num_inputs + j];
    }

    if (!isfinite(sum)) sum = 0;

    pre_activation[i] = sum;
    if (apply_relu)
        output[i] = d_ReLU(sum) * dropout_mask[i];
    else
        output[i] = sum;
}

__global__ void softmax_kernel(double* output, const double* input, int n,
                               double max_val, double sum_exp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double val = exp(input[i] - max_val) / sum_exp;
    if (val < 1e-15) val = 1e-15;
    if (val > 1 - 1e-15) val = 1 - 1e-15;
    output[i] = val;
}

__global__ void fc_backward_kernel(double* errors, const double* grad, const double* weights,
                                   const double* pre_activation, const double* dropout_mask,
                                   int num_neurons, int num_inputs, int is_output_layer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_neurons) return;

    double delta;
    if (is_output_layer) {
        delta = grad[i];
    } else {
        delta = grad[i] * d_ReLUDerivative(pre_activation[i]) * dropout_mask[i];
    }
    errors[i] = delta;
}

__global__ void fc_input_grad_kernel(double* input_grad, const double* errors,
                                     const double* weights, int num_neurons, int num_inputs) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_inputs) return;

    double sum = 0;
    for (int i = 0; i < num_neurons; i++) {
        sum += errors[i] * weights[i * num_inputs + j];
    }
    input_grad[j] = sum;
}

__global__ void pool_backward_kernel(double* input_grad, const double* grad,
                                     const int* max_indices_y, const int* max_indices_x,
                                     int channels, int input_h, int input_w,
                                     int output_h, int output_w, int pool_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * output_h * output_w;
    if (idx >= total) return;

    int c = idx / (output_h * output_w);
    int rem = idx % (output_h * output_w);
    int oh = rem / output_w;
    int ow = rem % output_w;

    int out_idx = c * output_h * output_w + oh * output_w + ow;
    int src_h = oh * pool_size + max_indices_y[out_idx];
    int src_w = ow * pool_size + max_indices_x[out_idx];
    int input_idx = c * input_h * input_w + src_h * input_w + src_w;

    atomicAddDouble(&input_grad[input_idx], grad[out_idx]);
}

__global__ void conv_weight_grad_kernel(double* weight_grads, const double* grad_with_relu,
                                        const double* padded_input,
                                        int num_filters, int input_channels, int kernel_size,
                                        int output_h, int output_w, int padded_h, int padded_w,
                                        int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = num_filters * input_channels * kernel_size * kernel_size;
    if (idx >= total_weights) return;

    int f = idx / (input_channels * kernel_size * kernel_size);
    int rem = idx % (input_channels * kernel_size * kernel_size);
    int c = rem / (kernel_size * kernel_size);
    rem = rem % (kernel_size * kernel_size);
    int kh = rem / kernel_size;
    int kw = rem % kernel_size;

    double w_grad = 0;
    for (int h = 0; h < output_h; h++) {
        for (int w = 0; w < output_w; w++) {
            int in_h = h * stride + kh;
            int in_w = w * stride + kw;
            int grad_idx = f * output_h * output_w + h * output_w + w;
            int input_idx = c * padded_h * padded_w + in_h * padded_w + in_w;
            w_grad += grad_with_relu[grad_idx] * padded_input[input_idx];
        }
    }
    weight_grads[idx] = w_grad;
}

__global__ void conv_bias_grad_kernel(double* bias_grads, const double* grad_with_relu,
                                      int num_filters, int output_h, int output_w) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_filters) return;

    double sum = 0;
    for (int h = 0; h < output_h; h++) {
        for (int w = 0; w < output_w; w++) {
            sum += grad_with_relu[f * output_h * output_w + h * output_w + w];
        }
    }
    bias_grads[f] = sum;
}

__global__ void apply_relu_deriv_kernel(double* grad_with_relu, const double* grad,
                                        const double* pre_activation, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    grad_with_relu[i] = grad[i] * d_ReLUDerivative(pre_activation[i]);
}

__global__ void conv_input_grad_kernel(double* input_grad, const double* grad_with_relu,
                                       const double* weights,
                                       int num_filters, int input_channels, int kernel_size,
                                       int input_h, int input_w, int output_h, int output_w,
                                       int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = input_channels * input_h * input_w;
    if (idx >= total) return;

    int c = idx / (input_h * input_w);
    int rem = idx % (input_h * input_w);
    int ih = rem / input_w;
    int iw = rem % input_w;

    double sum = 0;
    for (int f = 0; f < num_filters; f++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int oh = ih + padding - kh;
                int ow = iw + padding - kw;
                if (oh >= 0 && oh < output_h && ow >= 0 && ow < output_w &&
                    oh % stride == 0 && ow % stride == 0) {
                    oh /= stride;
                    ow /= stride;
                    int grad_idx = f * output_h * output_w + oh * output_w + ow;
                    int weight_idx = f * input_channels * kernel_size * kernel_size +
                                   c * kernel_size * kernel_size + kh * kernel_size + kw;
                    sum += grad_with_relu[grad_idx] * weights[weight_idx];
                }
            }
        }
    }
    input_grad[idx] = sum;
}

__global__ void adam_update_kernel(double* weights, double* m, double* v,
                                   const double* grads, double learning_rate,
                                   double beta1, double beta2, int timestep, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double grad = d_ClipGrad(grads[i]);
    m[i] = beta1 * m[i] + (1 - beta1) * grad;
    v[i] = beta2 * v[i] + (1 - beta2) * grad * grad;

    double m_hat = m[i] / (1 - pow(beta1, (double)timestep));
    double v_hat = v[i] / (1 - pow(beta2, (double)timestep));
    double update = learning_rate * m_hat / (sqrt(v_hat) + 1e-8);

    if (isfinite(update))
        weights[i] -= update;
}

__global__ void zero_array_kernel(double* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = 0;
}

__global__ void pad_input_kernel(double* padded, const double* input,
                                 int channels, int height, int width, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int padded_h = height + 2 * padding;
    int padded_w = width + 2 * padding;
    int total = channels * padded_h * padded_w;
    if (idx >= total) return;

    int c = idx / (padded_h * padded_w);
    int rem = idx % (padded_h * padded_w);
    int ph = rem / padded_w;
    int pw = rem % padded_w;

    int src_h = ph - padding;
    int src_w = pw - padding;

    if (src_h >= 0 && src_h < height && src_w >= 0 && src_w < width) {
        padded[idx] = input[c * height * width + src_h * width + src_w];
    } else {
        padded[idx] = 0;
    }
}

}
"#;

struct ConvLayerGPU {
    d_weights: CudaSlice<f64>,
    d_biases: CudaSlice<f64>,
    d_weights_m: CudaSlice<f64>,
    d_weights_v: CudaSlice<f64>,
    d_bias_m: CudaSlice<f64>,
    d_bias_v: CudaSlice<f64>,
    d_weight_grads: CudaSlice<f64>,
    d_bias_grads: CudaSlice<f64>,
    d_output: CudaSlice<f64>,
    d_pre_activation: CudaSlice<f64>,
    d_padded_input: CudaSlice<f64>,
    num_filters: i32,
    input_channels: i32,
    kernel_size: i32,
    stride: i32,
    padding: i32,
    output_h: i32,
    output_w: i32,
}

#[allow(dead_code)]
struct PoolLayerGPU {
    d_output: CudaSlice<f64>,
    d_max_indices_y: CudaSlice<i32>,
    d_max_indices_x: CudaSlice<i32>,
    pool_size: i32,
    stride: i32,
    output_h: i32,
    output_w: i32,
}

struct FCLayerGPU {
    d_weights: CudaSlice<f64>,
    d_biases: CudaSlice<f64>,
    d_weights_m: CudaSlice<f64>,
    d_weights_v: CudaSlice<f64>,
    d_bias_m: CudaSlice<f64>,
    d_bias_v: CudaSlice<f64>,
    d_output: CudaSlice<f64>,
    d_pre_activation: CudaSlice<f64>,
    d_errors: CudaSlice<f64>,
    d_dropout_mask: CudaSlice<f64>,
    num_neurons: i32,
    num_inputs: i32,
}

#[derive(Serialize, Deserialize)]
struct ModelJson {
    input_width: i32,
    input_height: i32,
    input_channels: i32,
    output_size: i32,
    conv_filters: Vec<i32>,
    kernel_sizes: Vec<i32>,
    pool_sizes: Vec<i32>,
    fc_layer_sizes: Vec<i32>,
    learning_rate: f64,
    dropout_rate: f64,
    activation: String,
    output_activation: String,
    loss_type: String,
    gradient_clip: f64,
    conv_layers: Vec<ConvLayerJson>,
    pool_layers: Vec<PoolLayerJson>,
    fc_layers: Vec<FCLayerJson>,
    output_layer: FCLayerJson,
}

#[derive(Serialize, Deserialize)]
struct ConvLayerJson {
    filters: Vec<FilterJson>,
}

#[derive(Serialize, Deserialize)]
struct FilterJson {
    bias: f64,
    weights: Vec<Vec<Vec<f64>>>,
}

#[derive(Serialize, Deserialize)]
struct PoolLayerJson {
    #[serde(rename = "poolSize")]
    pool_size: i32,
}

#[derive(Serialize, Deserialize)]
struct FCLayerJson {
    neurons: Vec<NeuronJson>,
}

#[derive(Serialize, Deserialize)]
struct NeuronJson {
    bias: f64,
    weights: Vec<f64>,
}

#[allow(dead_code)]
pub struct ConvolutionalNeuralNetworkCUDA {
    device: Arc<CudaDevice>,
    learning_rate: f64,
    dropout_rate: f64,
    gradient_clip: f64,
    beta1: f64,
    beta2: f64,
    adam_t: i32,
    is_training: bool,
    hidden_activation: ActivationType,
    output_activation: ActivationType,
    loss_function: LossType,

    conv_layers: Vec<ConvLayerGPU>,
    pool_layers: Vec<PoolLayerGPU>,
    fc_layers: Vec<FCLayerGPU>,
    output_layer: Option<FCLayerGPU>,

    input_width: i32,
    input_height: i32,
    input_channels: i32,
    flattened_size: i32,
    last_conv_h: i32,
    last_conv_w: i32,
    last_conv_c: i32,
    output_size: i32,

    d_flattened_features: CudaSlice<f64>,
    d_conv_grad: CudaSlice<f64>,
    d_fc_grad: CudaSlice<f64>,
    d_logits: CudaSlice<f64>,
    d_softmax_output: CudaSlice<f64>,

    max_neurons: i32,

    f_conv_filters: Vec<i32>,
    f_kernel_sizes: Vec<i32>,
    f_pool_sizes: Vec<i32>,
    f_fc_sizes: Vec<i32>,

    pub use_batch_norm: bool,
    pub batch_norm_params: Vec<BatchNormParams>,
}

fn compile_kernels(device: &Arc<CudaDevice>) -> Result<(), Box<dyn std::error::Error>> {
    let ptx = cudarc::nvrtc::compile_ptx(CUDA_KERNEL_SRC)?;
    device.load_ptx(ptx, "cnn_kernels", &[
        "conv_forward_kernel",
        "pool_forward_kernel",
        "fc_forward_kernel",
        "softmax_kernel",
        "fc_backward_kernel",
        "fc_input_grad_kernel",
        "pool_backward_kernel",
        "conv_weight_grad_kernel",
        "conv_bias_grad_kernel",
        "apply_relu_deriv_kernel",
        "conv_input_grad_kernel",
        "adam_update_kernel",
        "zero_array_kernel",
        "pad_input_kernel",
    ])?;
    Ok(())
}

fn launch_cfg(n: u32) -> LaunchConfig {
    let blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl ConvolutionalNeuralNetworkCUDA {
    pub fn new(
        input_width: i32,
        input_height: i32,
        input_channels: i32,
        conv_filters: &[i32],
        kernel_sizes: &[i32],
        pool_sizes: &[i32],
        fc_sizes: &[i32],
        output_size: i32,
        hidden_act: ActivationType,
        output_act: ActivationType,
        loss_type: LossType,
        learning_rate: f64,
        gradient_clip: f64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = CudaDevice::new(0)?;
        compile_kernels(&device)?;

        let mut rng = rand::thread_rng();

        let mut current_w = input_width;
        let mut current_h = input_height;
        let mut current_c = input_channels;

        let mut conv_layers_gpu = Vec::new();
        let mut pool_layers_gpu = Vec::new();

        for i in 0..conv_filters.len() {
            let kernel_padding = kernel_sizes[i] / 2;
            let output_h = (current_h + 2 * kernel_padding - kernel_sizes[i]) / 1 + 1;
            let output_w = (current_w + 2 * kernel_padding - kernel_sizes[i]) / 1 + 1;

            let weight_size = (conv_filters[i] * current_c * kernel_sizes[i] * kernel_sizes[i]) as usize;
            let output_sz = (conv_filters[i] * output_h * output_w) as usize;
            let padded_size = (current_c * (current_h + 2 * kernel_padding) * (current_w + 2 * kernel_padding)) as usize;

            let scale = (2.0 / (current_c * kernel_sizes[i] * kernel_sizes[i]) as f64).sqrt();
            let weights: Vec<f64> = (0..weight_size).map(|_| (rng.gen::<f64>() - 0.5) * scale).collect();

            let conv = ConvLayerGPU {
                d_weights: device.htod_sync_copy(&weights)?,
                d_biases: device.alloc_zeros(conv_filters[i] as usize)?,
                d_weights_m: device.alloc_zeros(weight_size)?,
                d_weights_v: device.alloc_zeros(weight_size)?,
                d_bias_m: device.alloc_zeros(conv_filters[i] as usize)?,
                d_bias_v: device.alloc_zeros(conv_filters[i] as usize)?,
                d_weight_grads: device.alloc_zeros(weight_size)?,
                d_bias_grads: device.alloc_zeros(conv_filters[i] as usize)?,
                d_output: device.alloc_zeros(output_sz)?,
                d_pre_activation: device.alloc_zeros(output_sz)?,
                d_padded_input: device.alloc_zeros(padded_size)?,
                num_filters: conv_filters[i],
                input_channels: current_c,
                kernel_size: kernel_sizes[i],
                stride: 1,
                padding: kernel_padding,
                output_h,
                output_w,
            };
            conv_layers_gpu.push(conv);

            current_w = output_w;
            current_h = output_h;
            current_c = conv_filters[i];

            if i < pool_sizes.len() {
                let pool_out_h = current_h / pool_sizes[i];
                let pool_out_w = current_w / pool_sizes[i];
                let pool_output_sz = (current_c * pool_out_h * pool_out_w) as usize;

                let pool = PoolLayerGPU {
                    d_output: device.alloc_zeros(pool_output_sz)?,
                    d_max_indices_y: device.alloc_zeros(pool_output_sz)?,
                    d_max_indices_x: device.alloc_zeros(pool_output_sz)?,
                    pool_size: pool_sizes[i],
                    stride: pool_sizes[i],
                    output_h: pool_out_h,
                    output_w: pool_out_w,
                };
                pool_layers_gpu.push(pool);

                current_w = pool_out_w;
                current_h = pool_out_h;
            }
        }

        let last_conv_h = current_h;
        let last_conv_w = current_w;
        let last_conv_c = current_c;
        let flattened_size = current_w * current_h * current_c;

        let mut fc_layers_gpu = Vec::new();
        let mut num_inputs = flattened_size;

        for &fc_size in fc_sizes {
            let weight_size = (fc_size * num_inputs) as usize;
            let scale = (2.0 / num_inputs as f64).sqrt();
            let weights: Vec<f64> = (0..weight_size).map(|_| (rng.gen::<f64>() - 0.5) * scale).collect();
            let mask: Vec<f64> = vec![1.0; fc_size as usize];

            let fc = FCLayerGPU {
                d_weights: device.htod_sync_copy(&weights)?,
                d_biases: device.alloc_zeros(fc_size as usize)?,
                d_weights_m: device.alloc_zeros(weight_size)?,
                d_weights_v: device.alloc_zeros(weight_size)?,
                d_bias_m: device.alloc_zeros(fc_size as usize)?,
                d_bias_v: device.alloc_zeros(fc_size as usize)?,
                d_output: device.alloc_zeros(fc_size as usize)?,
                d_pre_activation: device.alloc_zeros(fc_size as usize)?,
                d_errors: device.alloc_zeros(fc_size as usize)?,
                d_dropout_mask: device.htod_sync_copy(&mask)?,
                num_neurons: fc_size,
                num_inputs,
            };
            fc_layers_gpu.push(fc);
            num_inputs = fc_size;
        }

        let out_weight_size = (output_size * num_inputs) as usize;
        let scale = (2.0 / num_inputs as f64).sqrt();
        let out_weights: Vec<f64> = (0..out_weight_size).map(|_| (rng.gen::<f64>() - 0.5) * scale).collect();
        let out_mask: Vec<f64> = vec![1.0; output_size as usize];

        let output_layer = FCLayerGPU {
            d_weights: device.htod_sync_copy(&out_weights)?,
            d_biases: device.alloc_zeros(output_size as usize)?,
            d_weights_m: device.alloc_zeros(out_weight_size)?,
            d_weights_v: device.alloc_zeros(out_weight_size)?,
            d_bias_m: device.alloc_zeros(output_size as usize)?,
            d_bias_v: device.alloc_zeros(output_size as usize)?,
            d_output: device.alloc_zeros(output_size as usize)?,
            d_pre_activation: device.alloc_zeros(output_size as usize)?,
            d_errors: device.alloc_zeros(output_size as usize)?,
            d_dropout_mask: device.htod_sync_copy(&out_mask)?,
            num_neurons: output_size,
            num_inputs,
        };

        let mut max_neurons = flattened_size;
        for &fc_size in fc_sizes {
            if fc_size > max_neurons {
                max_neurons = fc_size;
            }
        }
        if output_size > max_neurons {
            max_neurons = output_size;
        }

        Ok(Self {
            device: device.clone(),
            learning_rate,
            dropout_rate: 0.0,
            gradient_clip,
            beta1: 0.9,
            beta2: 0.999,
            adam_t: 0,
            is_training: false,
            hidden_activation: hidden_act,
            output_activation: output_act,
            loss_function: loss_type,
            conv_layers: conv_layers_gpu,
            pool_layers: pool_layers_gpu,
            fc_layers: fc_layers_gpu,
            output_layer: Some(output_layer),
            input_width,
            input_height,
            input_channels,
            flattened_size,
            last_conv_h,
            last_conv_w,
            last_conv_c,
            output_size,
            d_flattened_features: device.alloc_zeros(flattened_size as usize)?,
            d_conv_grad: device.alloc_zeros(flattened_size as usize)?,
            d_fc_grad: device.alloc_zeros(max_neurons as usize)?,
            d_logits: device.alloc_zeros(output_size as usize)?,
            d_softmax_output: device.alloc_zeros(output_size as usize)?,
            max_neurons,
            f_conv_filters: conv_filters.to_vec(),
            f_kernel_sizes: kernel_sizes.to_vec(),
            f_pool_sizes: pool_sizes.to_vec(),
            f_fc_sizes: fc_sizes.to_vec(),
            use_batch_norm: false,
            batch_norm_params: Vec::new(),
        })
    }

    pub fn predict(&mut self, image_data: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        self.is_training = false;

        let d_input = self.device.htod_sync_copy(image_data)?;

        let mut current_h = self.input_height;
        let mut current_w = self.input_width;
        let mut current_c = self.input_channels;

        let pad_kernel = self.device.get_func("cnn_kernels", "pad_input_kernel").unwrap();
        let conv_kernel = self.device.get_func("cnn_kernels", "conv_forward_kernel").unwrap();
        let pool_kernel = self.device.get_func("cnn_kernels", "pool_forward_kernel").unwrap();
        let fc_kernel = self.device.get_func("cnn_kernels", "fc_forward_kernel").unwrap();
        let softmax_kernel_fn = self.device.get_func("cnn_kernels", "softmax_kernel").unwrap();

        let mut current_input_ptr = *d_input.device_ptr();

        for i in 0..self.conv_layers.len() {
            let conv = &self.conv_layers[i];
            let padded_h = current_h + 2 * conv.padding;
            let padded_w = current_w + 2 * conv.padding;
            let padded_size = (current_c * padded_h * padded_w) as u32;

            let padded_ptr = *conv.d_padded_input.device_ptr();
            unsafe {
                pad_kernel.clone().launch(
                    launch_cfg(padded_size),
                    (padded_ptr, current_input_ptr, current_c, current_h, current_w, conv.padding),
                )?;
            }

            let output_size = (conv.num_filters * conv.output_h * conv.output_w) as u32;

            let out_ptr = *conv.d_output.device_ptr();
            let pre_ptr = *conv.d_pre_activation.device_ptr();
            let wgt_ptr = *conv.d_weights.device_ptr();
            let bias_ptr = *conv.d_biases.device_ptr();

            let mut params: Vec<*mut c_void> = vec![
                &out_ptr as *const _ as *mut c_void,
                &pre_ptr as *const _ as *mut c_void,
                &padded_ptr as *const _ as *mut c_void,
                &wgt_ptr as *const _ as *mut c_void,
                &bias_ptr as *const _ as *mut c_void,
                &conv.input_channels as *const _ as *mut c_void,
                &conv.kernel_size as *const _ as *mut c_void,
                &current_h as *const _ as *mut c_void,
                &current_w as *const _ as *mut c_void,
                &conv.output_h as *const _ as *mut c_void,
                &conv.output_w as *const _ as *mut c_void,
                &conv.stride as *const _ as *mut c_void,
                &conv.padding as *const _ as *mut c_void,
                &conv.num_filters as *const _ as *mut c_void,
            ];
            unsafe {
                conv_kernel.clone().launch(launch_cfg(output_size), &mut params)?;
            }

            current_h = conv.output_h;
            current_w = conv.output_w;
            current_c = conv.num_filters;

            if i < self.pool_layers.len() {
                let pool = &self.pool_layers[i];
                let pool_output_size = (current_c * pool.output_h * pool.output_w) as u32;

                let pool_out_ptr = *pool.d_output.device_ptr();
                let pool_idx_y_ptr = *pool.d_max_indices_y.device_ptr();
                let pool_idx_x_ptr = *pool.d_max_indices_x.device_ptr();
                let conv_out_ptr = *self.conv_layers[i].d_output.device_ptr();

                unsafe {
                    pool_kernel.clone().launch(
                        launch_cfg(pool_output_size),
                        (
                            pool_out_ptr, pool_idx_y_ptr, pool_idx_x_ptr,
                            conv_out_ptr,
                            current_c, current_h, current_w,
                            pool.output_h, pool.output_w, pool.pool_size,
                        ),
                    )?;
                }
                current_h = pool.output_h;
                current_w = pool.output_w;
                current_input_ptr = *self.pool_layers[i].d_output.device_ptr();
            } else {
                current_input_ptr = *self.conv_layers[i].d_output.device_ptr();
            }
        }

        self.device.dtod_copy(&self.conv_layers.last().map(|c| c.d_output.slice(..)).unwrap_or(d_input.slice(..)), &mut self.d_flattened_features)?;

        let mut fc_input_ptr = *self.d_flattened_features.device_ptr();

        for i in 0..self.fc_layers.len() {
            let fc = &self.fc_layers[i];
            let out_ptr = *fc.d_output.device_ptr();
            let pre_ptr = *fc.d_pre_activation.device_ptr();
            let wgt_ptr = *fc.d_weights.device_ptr();
            let bias_ptr = *fc.d_biases.device_ptr();
            let mask_ptr = *fc.d_dropout_mask.device_ptr();
            let apply_relu: i32 = 1;

            unsafe {
                fc_kernel.clone().launch(
                    launch_cfg(fc.num_neurons as u32),
                    (
                        out_ptr, pre_ptr, fc_input_ptr,
                        wgt_ptr, bias_ptr, mask_ptr,
                        fc.num_neurons, fc.num_inputs, apply_relu,
                    ),
                )?;
            }
            fc_input_ptr = *self.fc_layers[i].d_output.device_ptr();
        }

        let output_layer = self.output_layer.as_ref().unwrap();
        let out_logits_ptr = *self.d_logits.device_ptr();
        let out_pre_ptr = *output_layer.d_pre_activation.device_ptr();
        let out_wgt_ptr = *output_layer.d_weights.device_ptr();
        let out_bias_ptr = *output_layer.d_biases.device_ptr();
        let out_mask_ptr = *output_layer.d_dropout_mask.device_ptr();
        let apply_relu: i32 = 0;

        unsafe {
            fc_kernel.clone().launch(
                launch_cfg(output_layer.num_neurons as u32),
                (
                    out_logits_ptr, out_pre_ptr, fc_input_ptr,
                    out_wgt_ptr, out_bias_ptr, out_mask_ptr,
                    output_layer.num_neurons, output_layer.num_inputs, apply_relu,
                ),
            )?;
        }

        self.device.synchronize()?;

        let h_logits = self.device.dtoh_sync_copy(&self.d_logits)?;

        let max_val = h_logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = h_logits.iter().map(|x| (x - max_val).exp()).sum();

        let softmax_out_ptr = *self.d_softmax_output.device_ptr();
        let logits_ptr = *self.d_logits.device_ptr();

        unsafe {
            softmax_kernel_fn.clone().launch(
                launch_cfg(self.output_size as u32),
                (softmax_out_ptr, logits_ptr, self.output_size, max_val, sum_exp),
            )?;
        }

        self.device.synchronize()?;

        let result = self.device.dtoh_sync_copy(&self.d_softmax_output)?;
        Ok(result)
    }

    pub fn train_step(&mut self, image_data: &[f64], target: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
        self.is_training = true;

        let prediction = self.predict(image_data)?;

        let output_grad: Vec<f64> = prediction.iter().zip(target.iter()).map(|(p, t)| p - t).collect();
        self.device.htod_sync_copy_into(&output_grad, &mut self.d_fc_grad)?;

        let fc_backward_kernel = self.device.get_func("cnn_kernels", "fc_backward_kernel").unwrap();
        let fc_input_grad_kernel = self.device.get_func("cnn_kernels", "fc_input_grad_kernel").unwrap();
        let adam_update_kernel = self.device.get_func("cnn_kernels", "adam_update_kernel").unwrap();

        let output_layer = self.output_layer.as_ref().unwrap();
        unsafe {
            fc_backward_kernel.clone().launch(
                launch_cfg(output_layer.num_neurons as u32),
                (
                    &output_layer.d_errors, &self.d_fc_grad, &output_layer.d_weights,
                    &output_layer.d_pre_activation, &output_layer.d_dropout_mask,
                    output_layer.num_neurons, output_layer.num_inputs, 1i32,
                ),
            )?;
        }

        let prev_size = if self.fc_layers.is_empty() {
            self.flattened_size
        } else {
            self.fc_layers.last().unwrap().num_neurons
        };

        let mut fc_grad_temp: CudaSlice<f64> = self.device.alloc_zeros(prev_size as usize)?;
        unsafe {
            fc_input_grad_kernel.clone().launch(
                launch_cfg(prev_size as u32),
                (
                    &fc_grad_temp, &output_layer.d_errors, &output_layer.d_weights,
                    output_layer.num_neurons, prev_size,
                ),
            )?;
        }

        for i in (0..self.fc_layers.len()).rev() {
            let fc = &self.fc_layers[i];
            unsafe {
                fc_backward_kernel.clone().launch(
                    launch_cfg(fc.num_neurons as u32),
                    (
                        &fc.d_errors, &fc_grad_temp, &fc.d_weights,
                        &fc.d_pre_activation, &fc.d_dropout_mask,
                        fc.num_neurons, fc.num_inputs, 0i32,
                    ),
                )?;
            }

            let prev_sz = if i > 0 {
                self.fc_layers[i - 1].num_neurons
            } else {
                self.flattened_size
            };

            let new_grad: CudaSlice<f64> = self.device.alloc_zeros(prev_sz as usize)?;
            unsafe {
                fc_input_grad_kernel.clone().launch(
                    launch_cfg(prev_sz as u32),
                    (&new_grad, &fc.d_errors, &fc.d_weights, fc.num_neurons, prev_sz),
                )?;
            }
            fc_grad_temp = new_grad;
        }

        self.device.dtod_copy(&fc_grad_temp, &mut self.d_conv_grad)?;

        self.device.synchronize()?;
        self.adam_t += 1;

        for conv in &self.conv_layers {
            let weight_size = (conv.num_filters * conv.input_channels * conv.kernel_size * conv.kernel_size) as usize;
            unsafe {
                adam_update_kernel.clone().launch(
                    launch_cfg(weight_size as u32),
                    (
                        &conv.d_weights, &conv.d_weights_m, &conv.d_weights_v,
                        &conv.d_weight_grads, self.learning_rate,
                        self.beta1, self.beta2, self.adam_t, weight_size as i32,
                    ),
                )?;
                adam_update_kernel.clone().launch(
                    launch_cfg(conv.num_filters as u32),
                    (
                        &conv.d_biases, &conv.d_bias_m, &conv.d_bias_v,
                        &conv.d_bias_grads, self.learning_rate,
                        self.beta1, self.beta2, self.adam_t, conv.num_filters,
                    ),
                )?;
            }
        }

        for i in 0..self.fc_layers.len() {
            let fc = &self.fc_layers[i];
            let weight_size = (fc.num_neurons * fc.num_inputs) as usize;

            let h_errors = self.device.dtoh_sync_copy(&fc.d_errors)?;
            let prev_output = if i == 0 {
                self.device.dtoh_sync_copy(&self.d_flattened_features)?
            } else {
                self.device.dtoh_sync_copy(&self.fc_layers[i - 1].d_output)?
            };

            let mut h_grads = vec![0.0; weight_size];
            for n in 0..fc.num_neurons as usize {
                for inp in 0..fc.num_inputs as usize {
                    h_grads[n * fc.num_inputs as usize + inp] = h_errors[n] * prev_output[inp];
                }
            }

            let d_grads = self.device.htod_sync_copy(&h_grads)?;
            unsafe {
                adam_update_kernel.clone().launch(
                    launch_cfg(weight_size as u32),
                    (
                        &fc.d_weights, &fc.d_weights_m, &fc.d_weights_v,
                        &d_grads, self.learning_rate,
                        self.beta1, self.beta2, self.adam_t, weight_size as i32,
                    ),
                )?;
                adam_update_kernel.clone().launch(
                    launch_cfg(fc.num_neurons as u32),
                    (
                        &fc.d_biases, &fc.d_bias_m, &fc.d_bias_v,
                        &fc.d_errors, self.learning_rate,
                        self.beta1, self.beta2, self.adam_t, fc.num_neurons,
                    ),
                )?;
            }
        }

        {
            let output_layer = self.output_layer.as_ref().unwrap();
            let weight_size = (output_layer.num_neurons * output_layer.num_inputs) as usize;

            let h_errors = self.device.dtoh_sync_copy(&output_layer.d_errors)?;
            let prev_output = if self.fc_layers.is_empty() {
                self.device.dtoh_sync_copy(&self.d_flattened_features)?
            } else {
                self.device.dtoh_sync_copy(&self.fc_layers.last().unwrap().d_output)?
            };

            let mut h_grads = vec![0.0; weight_size];
            for n in 0..output_layer.num_neurons as usize {
                for inp in 0..output_layer.num_inputs as usize {
                    h_grads[n * output_layer.num_inputs as usize + inp] = h_errors[n] * prev_output[inp];
                }
            }

            let d_grads = self.device.htod_sync_copy(&h_grads)?;
            unsafe {
                adam_update_kernel.clone().launch(
                    launch_cfg(weight_size as u32),
                    (
                        &output_layer.d_weights, &output_layer.d_weights_m, &output_layer.d_weights_v,
                        &d_grads, self.learning_rate,
                        self.beta1, self.beta2, self.adam_t, weight_size as i32,
                    ),
                )?;
                adam_update_kernel.clone().launch(
                    launch_cfg(output_layer.num_neurons as u32),
                    (
                        &output_layer.d_biases, &output_layer.d_bias_m, &output_layer.d_bias_v,
                        &output_layer.d_errors, self.learning_rate,
                        self.beta1, self.beta2, self.adam_t, output_layer.num_neurons,
                    ),
                )?;
            }
        }

        self.device.synchronize()?;

        let mut loss = 0.0;
        for i in 0..self.output_size as usize {
            if target[i] > 0.0 {
                let mut p = prediction[i];
                if p < 1e-15 { p = 1e-15; }
                if p > 1.0 - 1e-15 { p = 1.0 - 1e-15; }
                loss -= target[i] * p.ln();
            }
        }

        Ok(loss)
    }

    pub fn save_to_json(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let activation_to_string = |a: ActivationType| -> String {
            match a {
                ActivationType::ReLU => "relu".to_string(),
                ActivationType::Sigmoid => "sigmoid".to_string(),
                ActivationType::Tanh => "tanh".to_string(),
                ActivationType::Linear => "linear".to_string(),
            }
        };

        let loss_to_string = |l: LossType| -> String {
            match l {
                LossType::MSE => "mse".to_string(),
                LossType::CrossEntropy => "crossentropy".to_string(),
            }
        };

        let mut conv_layers_json = Vec::new();
        for conv in &self.conv_layers {
            let h_weights = self.device.dtoh_sync_copy(&conv.d_weights)?;
            let h_biases = self.device.dtoh_sync_copy(&conv.d_biases)?;

            let mut filters = Vec::new();
            for f in 0..conv.num_filters as usize {
                let mut weights_3d = Vec::new();
                for c in 0..conv.input_channels as usize {
                    let mut weights_2d = Vec::new();
                    for ky in 0..conv.kernel_size as usize {
                        let mut weights_1d = Vec::new();
                        for kx in 0..conv.kernel_size as usize {
                            let idx = f * conv.input_channels as usize * conv.kernel_size as usize * conv.kernel_size as usize
                                + c * conv.kernel_size as usize * conv.kernel_size as usize
                                + ky * conv.kernel_size as usize + kx;
                            weights_1d.push(h_weights[idx]);
                        }
                        weights_2d.push(weights_1d);
                    }
                    weights_3d.push(weights_2d);
                }
                filters.push(FilterJson {
                    bias: h_biases[f],
                    weights: weights_3d,
                });
            }
            conv_layers_json.push(ConvLayerJson { filters });
        }

        let pool_layers_json: Vec<PoolLayerJson> = self.pool_layers.iter()
            .map(|p| PoolLayerJson { pool_size: p.pool_size })
            .collect();

        let mut fc_layers_json = Vec::new();
        for fc in &self.fc_layers {
            let h_weights = self.device.dtoh_sync_copy(&fc.d_weights)?;
            let h_biases = self.device.dtoh_sync_copy(&fc.d_biases)?;

            let mut neurons = Vec::new();
            for j in 0..fc.num_neurons as usize {
                let start = j * fc.num_inputs as usize;
                let end = start + fc.num_inputs as usize;
                neurons.push(NeuronJson {
                    bias: h_biases[j],
                    weights: h_weights[start..end].to_vec(),
                });
            }
            fc_layers_json.push(FCLayerJson { neurons });
        }

        let output_layer = self.output_layer.as_ref().unwrap();
        let h_out_weights = self.device.dtoh_sync_copy(&output_layer.d_weights)?;
        let h_out_biases = self.device.dtoh_sync_copy(&output_layer.d_biases)?;

        let mut output_neurons = Vec::new();
        for j in 0..output_layer.num_neurons as usize {
            let start = j * output_layer.num_inputs as usize;
            let end = start + output_layer.num_inputs as usize;
            output_neurons.push(NeuronJson {
                bias: h_out_biases[j],
                weights: h_out_weights[start..end].to_vec(),
            });
        }
        let output_layer_json = FCLayerJson { neurons: output_neurons };

        let model = ModelJson {
            input_width: self.input_width,
            input_height: self.input_height,
            input_channels: self.input_channels,
            output_size: self.output_size,
            conv_filters: self.f_conv_filters.clone(),
            kernel_sizes: self.f_kernel_sizes.clone(),
            pool_sizes: self.f_pool_sizes.clone(),
            fc_layer_sizes: self.f_fc_sizes.clone(),
            learning_rate: self.learning_rate,
            dropout_rate: self.dropout_rate,
            activation: activation_to_string(self.hidden_activation),
            output_activation: activation_to_string(self.output_activation),
            loss_type: loss_to_string(self.loss_function),
            gradient_clip: self.gradient_clip,
            conv_layers: conv_layers_json,
            pool_layers: pool_layers_json,
            fc_layers: fc_layers_json,
            output_layer: output_layer_json,
        };

        let file = File::create(filename)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &model)?;

        Ok(())
    }

    pub fn get_input_width(&self) -> i32 { self.input_width }
    pub fn get_input_height(&self) -> i32 { self.input_height }
    pub fn get_input_channels(&self) -> i32 { self.input_channels }
    pub fn get_output_size(&self) -> i32 { self.output_size }
    pub fn get_learning_rate(&self) -> f64 { self.learning_rate }
    pub fn get_gradient_clip(&self) -> f64 { self.gradient_clip }
    pub fn get_num_conv_layers(&self) -> i32 { self.conv_layers.len() as i32 }
    pub fn get_num_fc_layers(&self) -> i32 { self.fc_layers.len() as i32 }
    pub fn get_use_batch_norm(&self) -> bool { self.use_batch_norm }
    pub fn get_batch_norm_params(&self) -> &[BatchNormParams] { &self.batch_norm_params }

    pub fn initialize_batch_norm(&mut self) {
        self.use_batch_norm = true;
        self.batch_norm_params.clear();
        for &num_filters in &self.f_conv_filters {
            let mut params = BatchNormParams::new();
            params.initialize(num_filters as usize);
            self.batch_norm_params.push(params);
        }
    }

    pub fn apply_batch_norm(&self, input: &[f64], layer_idx: usize) -> Vec<f64> {
        if !self.use_batch_norm || layer_idx >= self.batch_norm_params.len() {
            return input.to_vec();
        }

        let params = &self.batch_norm_params[layer_idx];
        let channel_size = input.len() / params.gamma.len();
        let mut output = vec![0.0; input.len()];

        for c in 0..params.gamma.len() {
            for i in 0..channel_size {
                let idx = c * channel_size + i;
                if idx < input.len() {
                    let normalized = (input[idx] - params.running_mean[c])
                        / (params.running_var[c] + params.epsilon).sqrt();
                    output[idx] = params.gamma[c] * normalized + params.beta[c];
                }
            }
        }
        output
    }

    pub fn export_to_onnx(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;
        let mut file = File::create(filename)?;

        file.write_all(b"ONNX")?;
        file.write_all(&1i32.to_le_bytes())?;

        file.write_all(&self.input_width.to_le_bytes())?;
        file.write_all(&self.input_height.to_le_bytes())?;
        file.write_all(&self.input_channels.to_le_bytes())?;
        file.write_all(&self.output_size.to_le_bytes())?;

        let bn_flag: i32 = if self.use_batch_norm { 1 } else { 0 };
        file.write_all(&bn_flag.to_le_bytes())?;

        let num_conv = self.f_conv_filters.len() as i32;
        file.write_all(&num_conv.to_le_bytes())?;
        for i in 0..num_conv as usize {
            file.write_all(&self.f_conv_filters[i].to_le_bytes())?;
            file.write_all(&self.f_kernel_sizes[i].to_le_bytes())?;
            file.write_all(&self.f_pool_sizes[i].to_le_bytes())?;
        }

        let num_fc = self.f_fc_sizes.len() as i32;
        file.write_all(&num_fc.to_le_bytes())?;
        for &size in &self.f_fc_sizes {
            file.write_all(&size.to_le_bytes())?;
        }

        for conv in &self.conv_layers {
            let weights = self.device.dtoh_sync_copy(&conv.d_weights)?;
            let biases = self.device.dtoh_sync_copy(&conv.d_biases)?;
            let weight_count = weights.len() as i32;
            file.write_all(&weight_count.to_le_bytes())?;
            for w in &weights {
                file.write_all(&w.to_le_bytes())?;
            }
            let bias_count = biases.len() as i32;
            file.write_all(&bias_count.to_le_bytes())?;
            for b in &biases {
                file.write_all(&b.to_le_bytes())?;
            }
        }

        for fc in &self.fc_layers {
            let weights = self.device.dtoh_sync_copy(&fc.d_weights)?;
            let biases = self.device.dtoh_sync_copy(&fc.d_biases)?;
            let weight_count = weights.len() as i32;
            file.write_all(&weight_count.to_le_bytes())?;
            for w in &weights {
                file.write_all(&w.to_le_bytes())?;
            }
            let bias_count = biases.len() as i32;
            file.write_all(&bias_count.to_le_bytes())?;
            for b in &biases {
                file.write_all(&b.to_le_bytes())?;
            }
        }

        if let Some(ref out_layer) = self.output_layer {
            let weights = self.device.dtoh_sync_copy(&out_layer.d_weights)?;
            let biases = self.device.dtoh_sync_copy(&out_layer.d_biases)?;
            let weight_count = weights.len() as i32;
            file.write_all(&weight_count.to_le_bytes())?;
            for w in &weights {
                file.write_all(&w.to_le_bytes())?;
            }
            let bias_count = biases.len() as i32;
            file.write_all(&bias_count.to_le_bytes())?;
            for b in &biases {
                file.write_all(&b.to_le_bytes())?;
            }
        }

        if self.use_batch_norm {
            for bn in &self.batch_norm_params {
                let size = bn.gamma.len() as i32;
                file.write_all(&size.to_le_bytes())?;
                for j in 0..bn.gamma.len() {
                    file.write_all(&bn.gamma[j].to_le_bytes())?;
                    file.write_all(&bn.beta[j].to_le_bytes())?;
                    file.write_all(&bn.running_mean[j].to_le_bytes())?;
                    file.write_all(&bn.running_var[j].to_le_bytes())?;
                }
            }
        }

        println!("Model exported to ONNX: {}", filename);
        Ok(())
    }

    pub fn import_from_onnx(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        use std::io::Read;
        let mut file = File::open(filename)?;

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"ONNX" {
            return Err("Invalid ONNX file format".into());
        }

        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)?;
        let _version = i32::from_le_bytes(buf4);

        file.read_exact(&mut buf4)?;
        let input_w = i32::from_le_bytes(buf4);
        file.read_exact(&mut buf4)?;
        let input_h = i32::from_le_bytes(buf4);
        file.read_exact(&mut buf4)?;
        let input_c = i32::from_le_bytes(buf4);
        file.read_exact(&mut buf4)?;
        let output_size = i32::from_le_bytes(buf4);

        file.read_exact(&mut buf4)?;
        let use_bn = i32::from_le_bytes(buf4) == 1;

        file.read_exact(&mut buf4)?;
        let num_conv = i32::from_le_bytes(buf4) as usize;

        let mut conv_filters = Vec::with_capacity(num_conv);
        let mut kernel_sizes = Vec::with_capacity(num_conv);
        let mut pool_sizes = Vec::with_capacity(num_conv);

        for _ in 0..num_conv {
            file.read_exact(&mut buf4)?;
            conv_filters.push(i32::from_le_bytes(buf4));
            file.read_exact(&mut buf4)?;
            kernel_sizes.push(i32::from_le_bytes(buf4));
            file.read_exact(&mut buf4)?;
            pool_sizes.push(i32::from_le_bytes(buf4));
        }

        file.read_exact(&mut buf4)?;
        let num_fc = i32::from_le_bytes(buf4) as usize;
        let mut fc_sizes = Vec::with_capacity(num_fc);
        for _ in 0..num_fc {
            file.read_exact(&mut buf4)?;
            fc_sizes.push(i32::from_le_bytes(buf4));
        }

        let mut cnn = Self::new(
            input_w, input_h, input_c,
            &conv_filters, &kernel_sizes, &pool_sizes, &fc_sizes,
            output_size,
            ActivationType::ReLU, ActivationType::Linear, LossType::CrossEntropy,
            0.001, 5.0
        )?;

        cnn.use_batch_norm = use_bn;

        for conv in &mut cnn.conv_layers {
            file.read_exact(&mut buf4)?;
            let weight_count = i32::from_le_bytes(buf4) as usize;
            let mut weights = vec![0.0f64; weight_count];
            let mut buf8 = [0u8; 8];
            for w in &mut weights {
                file.read_exact(&mut buf8)?;
                *w = f64::from_le_bytes(buf8);
            }
            cnn.device.htod_sync_copy_into(&weights, &mut conv.d_weights)?;

            file.read_exact(&mut buf4)?;
            let bias_count = i32::from_le_bytes(buf4) as usize;
            let mut biases = vec![0.0f64; bias_count];
            for b in &mut biases {
                file.read_exact(&mut buf8)?;
                *b = f64::from_le_bytes(buf8);
            }
            cnn.device.htod_sync_copy_into(&biases, &mut conv.d_biases)?;
        }

        for fc in &mut cnn.fc_layers {
            let mut buf8 = [0u8; 8];
            file.read_exact(&mut buf4)?;
            let weight_count = i32::from_le_bytes(buf4) as usize;
            let mut weights = vec![0.0f64; weight_count];
            for w in &mut weights {
                file.read_exact(&mut buf8)?;
                *w = f64::from_le_bytes(buf8);
            }
            cnn.device.htod_sync_copy_into(&weights, &mut fc.d_weights)?;

            file.read_exact(&mut buf4)?;
            let bias_count = i32::from_le_bytes(buf4) as usize;
            let mut biases = vec![0.0f64; bias_count];
            for b in &mut biases {
                file.read_exact(&mut buf8)?;
                *b = f64::from_le_bytes(buf8);
            }
            cnn.device.htod_sync_copy_into(&biases, &mut fc.d_biases)?;
        }

        if let Some(ref mut out_layer) = cnn.output_layer {
            let mut buf8 = [0u8; 8];
            file.read_exact(&mut buf4)?;
            let weight_count = i32::from_le_bytes(buf4) as usize;
            let mut weights = vec![0.0f64; weight_count];
            for w in &mut weights {
                file.read_exact(&mut buf8)?;
                *w = f64::from_le_bytes(buf8);
            }
            cnn.device.htod_sync_copy_into(&weights, &mut out_layer.d_weights)?;

            file.read_exact(&mut buf4)?;
            let bias_count = i32::from_le_bytes(buf4) as usize;
            let mut biases = vec![0.0f64; bias_count];
            for b in &mut biases {
                file.read_exact(&mut buf8)?;
                *b = f64::from_le_bytes(buf8);
            }
            cnn.device.htod_sync_copy_into(&biases, &mut out_layer.d_biases)?;
        }

        if use_bn {
            cnn.batch_norm_params.clear();
            for _ in 0..num_conv {
                let mut buf8 = [0u8; 8];
                file.read_exact(&mut buf4)?;
                let size = i32::from_le_bytes(buf4) as usize;
                let mut bn = BatchNormParams::new();
                bn.initialize(size);
                for j in 0..size {
                    file.read_exact(&mut buf8)?;
                    bn.gamma[j] = f64::from_le_bytes(buf8);
                    file.read_exact(&mut buf8)?;
                    bn.beta[j] = f64::from_le_bytes(buf8);
                    file.read_exact(&mut buf8)?;
                    bn.running_mean[j] = f64::from_le_bytes(buf8);
                    file.read_exact(&mut buf8)?;
                    bn.running_var[j] = f64::from_le_bytes(buf8);
                }
                cnn.batch_norm_params.push(bn);
            }
        }

        println!("Model imported from ONNX: {}", filename);
        Ok(cnn)
    }
}

