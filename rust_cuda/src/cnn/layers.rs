#![allow(dead_code)]

use cudarc::driver::CudaSlice;

pub struct ConvLayerGPU {
    pub d_weights: CudaSlice<f64>,
    pub d_biases: CudaSlice<f64>,
    pub d_weights_m: CudaSlice<f64>,
    pub d_weights_v: CudaSlice<f64>,
    pub d_bias_m: CudaSlice<f64>,
    pub d_bias_v: CudaSlice<f64>,
    pub d_weight_grads: CudaSlice<f64>,
    pub d_bias_grads: CudaSlice<f64>,
    pub d_output: CudaSlice<f64>,
    pub d_pre_activation: CudaSlice<f64>,
    pub d_padded_input: CudaSlice<f64>,
    pub num_filters: i32,
    pub input_channels: i32,
    pub kernel_size: i32,
    pub stride: i32,
    pub padding: i32,
    pub output_h: i32,
    pub output_w: i32,
}

pub struct PoolLayerGPU {
    pub d_output: CudaSlice<f64>,
    pub d_max_indices_y: CudaSlice<i32>,
    pub d_max_indices_x: CudaSlice<i32>,
    pub pool_size: i32,
    pub stride: i32,
    pub output_h: i32,
    pub output_w: i32,
}

pub struct FCLayerGPU {
    pub d_weights: CudaSlice<f64>,
    pub d_biases: CudaSlice<f64>,
    pub d_weights_m: CudaSlice<f64>,
    pub d_weights_v: CudaSlice<f64>,
    pub d_bias_m: CudaSlice<f64>,
    pub d_bias_v: CudaSlice<f64>,
    pub d_output: CudaSlice<f64>,
    pub d_pre_activation: CudaSlice<f64>,
    pub d_errors: CudaSlice<f64>,
    pub d_dropout_mask: CudaSlice<f64>,
    pub num_neurons: i32,
    pub num_inputs: i32,
}
