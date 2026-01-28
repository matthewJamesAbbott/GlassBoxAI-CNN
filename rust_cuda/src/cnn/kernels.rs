pub const CUDA_KERNELS: &str = r#"
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

// Conv forward with packed dimension params
// dims: [input_channels, kernel_size, input_h, input_w, output_h, output_w, stride, padding, num_filters]
__global__ void conv_forward_kernel(double* output, double* pre_activation,
                                     const double* input, const double* weights,
                                     const double* biases, const int* dims) {
    int input_channels = dims[0];
    int kernel_size = dims[1];
    int input_h = dims[2];
    int input_w = dims[3];
    int output_h = dims[4];
    int output_w = dims[5];
    int stride = dims[6];
    int padding = dims[7];
    int num_filters = dims[8];

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

// Pool forward with packed dims: [channels, input_h, input_w, output_h, output_w, pool_size]
__global__ void pool_forward_kernel(double* output, int* max_indices_y, int* max_indices_x,
                                     const double* input, const int* dims) {
    int channels = dims[0];
    int input_h = dims[1];
    int input_w = dims[2];
    int output_h = dims[3];
    int output_w = dims[4];
    int pool_size = dims[5];

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

// FC forward with packed dims: [num_neurons, num_inputs, apply_relu]
__global__ void fc_forward_kernel(double* output, double* pre_activation,
                                   const double* input, const double* weights,
                                   const double* biases, const double* dropout_mask,
                                   const int* dims) {
    int num_neurons = dims[0];
    int num_inputs = dims[1];
    int apply_relu = dims[2];

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

// FC backward with packed dims: [num_neurons, num_inputs, is_output_layer]
__global__ void fc_backward_kernel(double* errors, const double* grad,
                                    const double* weights,
                                    const double* pre_activation, const double* dropout_mask,
                                    const int* dims) {
    int num_neurons = dims[0];
    int is_output_layer = dims[2];

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

// FC input grad with dims: [num_neurons, num_inputs]
__global__ void fc_input_grad_kernel(double* input_grad, const double* errors,
                                      const double* weights, const int* dims) {
    int num_neurons = dims[0];
    int num_inputs = dims[1];

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_inputs) return;

    double sum = 0;
    for (int i = 0; i < num_neurons; i++) {
        sum += errors[i] * weights[i * num_inputs + j];
    }
    input_grad[j] = sum;
}

// Pool backward with dims: [channels, input_h, input_w, output_h, output_w, pool_size]
__global__ void pool_backward_kernel(double* input_grad, const double* grad,
                                      const int* max_indices_y, const int* max_indices_x,
                                      const int* dims) {
    int channels = dims[0];
    int input_h = dims[1];
    int input_w = dims[2];
    int output_h = dims[3];
    int output_w = dims[4];
    int pool_size = dims[5];

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

// Conv weight grad with dims: [num_filters, input_channels, kernel_size, output_h, output_w, padded_h, padded_w, stride]
__global__ void conv_weight_grad_kernel(double* weight_grads,
                                         const double* grad_with_relu, const double* padded_input,
                                         const int* dims) {
    int num_filters = dims[0];
    int input_channels = dims[1];
    int kernel_size = dims[2];
    int output_h = dims[3];
    int output_w = dims[4];
    int padded_h = dims[5];
    int padded_w = dims[6];
    int stride = dims[7];

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

// Conv bias grad with dims: [num_filters, output_h, output_w]
__global__ void conv_bias_grad_kernel(double* bias_grads, const double* grad_with_relu,
                                       const int* dims) {
    int num_filters = dims[0];
    int output_h = dims[1];
    int output_w = dims[2];

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

// Adam update with dims: [n, timestep]
__global__ void adam_update_kernel(double* weights, double* m, double* v,
                                    const double* grads, const double* params,
                                    const int* dims) {
    int n = dims[0];
    int timestep = dims[1];
    double learning_rate = params[0];
    double beta1 = params[1];
    double beta2 = params[2];

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

// Pad input with dims: [channels, height, width, padding]
__global__ void pad_input_kernel(double* padded, const double* input, const int* dims) {
    int channels = dims[0];
    int height = dims[1];
    int width = dims[2];
    int padding = dims[3];

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

pub const KERNEL_NAMES: &[&str] = &[
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
    "adam_update_kernel",
    "zero_array_kernel",
    "pad_input_kernel",
];
