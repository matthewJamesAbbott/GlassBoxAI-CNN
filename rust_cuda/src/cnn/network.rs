#![allow(dead_code)]

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use rand::Rng;
use std::fs::File;
use std::io::{Write, BufWriter};
use std::sync::Arc;

use super::types::{ActivationType, LossType, Loss, BatchNormParams, BLOCK_SIZE, MODEL_MAGIC};
use super::kernels::{CUDA_KERNELS, KERNEL_NAMES};
use super::layers::{ConvLayerGPU, PoolLayerGPU, FCLayerGPU};

pub struct ConvolutionalNeuralNetworkCUDA {
    device: Arc<CudaDevice>,
    pub learning_rate: f64,
    pub dropout_rate: f64,
    pub gradient_clip: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub adam_t: i32,
    pub is_training: bool,
    pub hidden_activation: ActivationType,
    pub output_activation: ActivationType,
    pub loss_function: LossType,

    pub conv_layers: Vec<ConvLayerGPU>,
    pub pool_layers: Vec<PoolLayerGPU>,
    pub fc_layers: Vec<FCLayerGPU>,
    pub output_layer: FCLayerGPU,

    pub input_width: i32,
    pub input_height: i32,
    pub input_channels: i32,
    pub flattened_size: i32,
    pub last_conv_h: i32,
    pub last_conv_w: i32,
    pub last_conv_c: i32,
    pub output_size: i32,

    pub d_flattened_features: CudaSlice<f64>,
    d_conv_grad: CudaSlice<f64>,
    d_fc_grad: CudaSlice<f64>,
    d_target: CudaSlice<f64>,
    d_logits: CudaSlice<f64>,
    d_softmax_output: CudaSlice<f64>,

    pub max_neurons: i32,

    pub f_conv_filters: Vec<i32>,
    pub f_kernel_sizes: Vec<i32>,
    pub f_pool_sizes: Vec<i32>,
    pub f_fc_sizes: Vec<i32>,

    pub use_batch_norm: bool,
    pub batch_norm_params: Vec<BatchNormParams>,
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

        let ptx = cudarc::nvrtc::compile_ptx(CUDA_KERNELS)?;
        device.load_ptx(ptx, "cnn_kernels", KERNEL_NAMES)?;

        let mut current_w = input_width;
        let mut current_h = input_height;
        let mut current_c = input_channels;

        let mut conv_layers = Vec::new();
        let mut pool_layers = Vec::new();

        for i in 0..conv_filters.len() {
            let kernel_padding = kernel_sizes[i] / 2;
            let conv = Self::allocate_conv_layer(
                &device,
                conv_filters[i],
                current_c,
                kernel_sizes[i],
                1,
                kernel_padding,
                current_h,
                current_w,
            )?;
            current_w = conv.output_w;
            current_h = conv.output_h;
            current_c = conv_filters[i];
            conv_layers.push(conv);

            if i < pool_sizes.len() {
                let pool = Self::allocate_pool_layer(
                    &device,
                    pool_sizes[i],
                    pool_sizes[i],
                    current_h,
                    current_w,
                    current_c,
                )?;
                current_w = pool.output_w;
                current_h = pool.output_h;
                pool_layers.push(pool);
            }
        }

        let last_conv_h = current_h;
        let last_conv_w = current_w;
        let last_conv_c = current_c;
        let flattened_size = current_w * current_h * current_c;

        let mut fc_layer_list = Vec::new();
        let mut num_inputs = flattened_size;

        for &size in fc_sizes {
            let fc = Self::allocate_fc_layer(&device, size, num_inputs)?;
            num_inputs = size;
            fc_layer_list.push(fc);
        }

        let output_layer = Self::allocate_fc_layer(&device, output_size, num_inputs)?;

        let mut max_neurons = flattened_size;
        for &size in fc_sizes {
            if size > max_neurons {
                max_neurons = size;
            }
        }
        if output_size > max_neurons {
            max_neurons = output_size;
        }

        let d_flattened_features = device.alloc_zeros::<f64>(flattened_size as usize)?;
        let d_conv_grad = device.alloc_zeros::<f64>(flattened_size as usize)?;
        let d_fc_grad = device.alloc_zeros::<f64>(max_neurons as usize)?;
        let d_target = device.alloc_zeros::<f64>(output_size as usize)?;
        let d_logits = device.alloc_zeros::<f64>(output_size as usize)?;
        let d_softmax_output = device.alloc_zeros::<f64>(output_size as usize)?;

        Ok(Self {
            device,
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
            conv_layers,
            pool_layers,
            fc_layers: fc_layer_list,
            output_layer,
            input_width,
            input_height,
            input_channels,
            flattened_size,
            last_conv_h,
            last_conv_w,
            last_conv_c,
            output_size,
            d_flattened_features,
            d_conv_grad,
            d_fc_grad,
            d_target,
            d_logits,
            d_softmax_output,
            max_neurons,
            f_conv_filters: conv_filters.to_vec(),
            f_kernel_sizes: kernel_sizes.to_vec(),
            f_pool_sizes: pool_sizes.to_vec(),
            f_fc_sizes: fc_sizes.to_vec(),
            use_batch_norm: false,
            batch_norm_params: Vec::new(),
        })
    }

    fn allocate_conv_layer(
        device: &Arc<CudaDevice>,
        num_filters: i32,
        input_channels: i32,
        kernel_size: i32,
        stride: i32,
        padding: i32,
        input_h: i32,
        input_w: i32,
    ) -> Result<ConvLayerGPU, Box<dyn std::error::Error>> {
        let output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
        let output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

        let weight_size = (num_filters * input_channels * kernel_size * kernel_size) as usize;
        let output_sz = (num_filters * output_h * output_w) as usize;
        let padded_size = (input_channels * (input_h + 2 * padding) * (input_w + 2 * padding)) as usize;

        let scale = (2.0 / (input_channels * kernel_size * kernel_size) as f64).sqrt();
        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = (0..weight_size)
            .map(|_| (rng.gen::<f64>() - 0.5) * scale)
            .collect();

        let d_weights = device.htod_copy(weights)?;
        let d_biases = device.alloc_zeros::<f64>(num_filters as usize)?;
        let d_weights_m = device.alloc_zeros::<f64>(weight_size)?;
        let d_weights_v = device.alloc_zeros::<f64>(weight_size)?;
        let d_bias_m = device.alloc_zeros::<f64>(num_filters as usize)?;
        let d_bias_v = device.alloc_zeros::<f64>(num_filters as usize)?;
        let d_weight_grads = device.alloc_zeros::<f64>(weight_size)?;
        let d_bias_grads = device.alloc_zeros::<f64>(num_filters as usize)?;
        let d_output = device.alloc_zeros::<f64>(output_sz)?;
        let d_pre_activation = device.alloc_zeros::<f64>(output_sz)?;
        let d_padded_input = device.alloc_zeros::<f64>(padded_size)?;

        Ok(ConvLayerGPU {
            d_weights,
            d_biases,
            d_weights_m,
            d_weights_v,
            d_bias_m,
            d_bias_v,
            d_weight_grads,
            d_bias_grads,
            d_output,
            d_pre_activation,
            d_padded_input,
            num_filters,
            input_channels,
            kernel_size,
            stride,
            padding,
            output_h,
            output_w,
        })
    }

    fn allocate_pool_layer(
        device: &Arc<CudaDevice>,
        pool_size: i32,
        stride: i32,
        input_h: i32,
        input_w: i32,
        channels: i32,
    ) -> Result<PoolLayerGPU, Box<dyn std::error::Error>> {
        let output_h = input_h / pool_size;
        let output_w = input_w / pool_size;
        let output_sz = (channels * output_h * output_w) as usize;

        let d_output = device.alloc_zeros::<f64>(output_sz)?;
        let d_max_indices_y = device.alloc_zeros::<i32>(output_sz)?;
        let d_max_indices_x = device.alloc_zeros::<i32>(output_sz)?;

        Ok(PoolLayerGPU {
            d_output,
            d_max_indices_y,
            d_max_indices_x,
            pool_size,
            stride,
            output_h,
            output_w,
        })
    }

    fn allocate_fc_layer(
        device: &Arc<CudaDevice>,
        num_neurons: i32,
        num_inputs: i32,
    ) -> Result<FCLayerGPU, Box<dyn std::error::Error>> {
        let weight_size = (num_neurons * num_inputs) as usize;

        let scale = (2.0 / num_inputs as f64).sqrt();
        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = (0..weight_size)
            .map(|_| (rng.gen::<f64>() - 0.5) * scale)
            .collect();

        let d_weights = device.htod_copy(weights)?;
        let d_biases = device.alloc_zeros::<f64>(num_neurons as usize)?;
        let d_weights_m = device.alloc_zeros::<f64>(weight_size)?;
        let d_weights_v = device.alloc_zeros::<f64>(weight_size)?;
        let d_bias_m = device.alloc_zeros::<f64>(num_neurons as usize)?;
        let d_bias_v = device.alloc_zeros::<f64>(num_neurons as usize)?;
        let d_output = device.alloc_zeros::<f64>(num_neurons as usize)?;
        let d_pre_activation = device.alloc_zeros::<f64>(num_neurons as usize)?;
        let d_errors = device.alloc_zeros::<f64>(num_neurons as usize)?;
        let dropout_mask: Vec<f64> = vec![1.0; num_neurons as usize];
        let d_dropout_mask = device.htod_copy(dropout_mask)?;

        Ok(FCLayerGPU {
            d_weights,
            d_biases,
            d_weights_m,
            d_weights_v,
            d_bias_m,
            d_bias_v,
            d_output,
            d_pre_activation,
            d_errors,
            d_dropout_mask,
            num_neurons,
            num_inputs,
        })
    }

    fn launch_config(n: u32) -> LaunchConfig {
        let blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    pub fn predict(&mut self, image_data: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        self.is_training = false;

        let d_input = self.device.htod_copy(image_data.to_vec())?;

        let mut current_h = self.input_height;
        let mut current_w = self.input_width;
        let mut current_c = self.input_channels;

        let pad_input_fn = self.device.get_func("cnn_kernels", "pad_input_kernel").unwrap();
        let conv_forward_fn = self.device.get_func("cnn_kernels", "conv_forward_kernel").unwrap();
        let pool_forward_fn = self.device.get_func("cnn_kernels", "pool_forward_kernel").unwrap();
        let fc_forward_fn = self.device.get_func("cnn_kernels", "fc_forward_kernel").unwrap();
        let softmax_fn = self.device.get_func("cnn_kernels", "softmax_kernel").unwrap();

        let mut current_input_ptr = d_input.clone();

        for i in 0..self.conv_layers.len() {
            let conv = &self.conv_layers[i];
            let padded_h = current_h + 2 * conv.padding;
            let padded_w = current_w + 2 * conv.padding;
            let padded_size = current_c * padded_h * padded_w;

            // Pad input
            let pad_dims = self.device.htod_copy(vec![current_c, current_h, current_w, conv.padding])?;
            let cfg = Self::launch_config(padded_size as u32);
            unsafe {
                pad_input_fn.clone().launch(cfg, (
                    &conv.d_padded_input,
                    &current_input_ptr,
                    &pad_dims,
                ))?;
            }

            // Conv forward
            let conv_dims = self.device.htod_copy(vec![
                conv.input_channels, conv.kernel_size, current_h, current_w,
                conv.output_h, conv.output_w, conv.stride, conv.padding, conv.num_filters
            ])?;
            let output_size = conv.num_filters * conv.output_h * conv.output_w;
            let cfg = Self::launch_config(output_size as u32);
            unsafe {
                conv_forward_fn.clone().launch(cfg, (
                    &conv.d_output,
                    &conv.d_pre_activation,
                    &conv.d_padded_input,
                    &conv.d_weights,
                    &conv.d_biases,
                    &conv_dims,
                ))?;
            }

            current_h = conv.output_h;
            current_w = conv.output_w;
            current_c = conv.num_filters;

            if i < self.pool_layers.len() {
                let pool = &self.pool_layers[i];
                let pool_dims = self.device.htod_copy(vec![
                    current_c, current_h, current_w,
                    pool.output_h, pool.output_w, pool.pool_size
                ])?;
                let pool_output_size = current_c * pool.output_h * pool.output_w;
                let cfg = Self::launch_config(pool_output_size as u32);
                unsafe {
                    pool_forward_fn.clone().launch(cfg, (
                        &pool.d_output,
                        &pool.d_max_indices_y,
                        &pool.d_max_indices_x,
                        &conv.d_output,
                        &pool_dims,
                    ))?;
                }
                current_h = pool.output_h;
                current_w = pool.output_w;
                current_input_ptr = pool.d_output.clone();
            } else {
                current_input_ptr = conv.d_output.clone();
            }
        }

        self.device.dtod_copy(&current_input_ptr, &mut self.d_flattened_features)?;

        // FC layers
        let mut fc_input_ptr = self.d_flattened_features.clone();
        for fc in &self.fc_layers {
            let fc_dims = self.device.htod_copy(vec![fc.num_neurons, fc.num_inputs, 1])?;
            let cfg = Self::launch_config(fc.num_neurons as u32);
            unsafe {
                fc_forward_fn.clone().launch(cfg, (
                    &fc.d_output,
                    &fc.d_pre_activation,
                    &fc_input_ptr,
                    &fc.d_weights,
                    &fc.d_biases,
                    &fc.d_dropout_mask,
                    &fc_dims,
                ))?;
            }
            fc_input_ptr = fc.d_output.clone();
        }

        // Output layer
        let out_dims = self.device.htod_copy(vec![self.output_layer.num_neurons, self.output_layer.num_inputs, 0])?;
        let cfg = Self::launch_config(self.output_layer.num_neurons as u32);
        unsafe {
            fc_forward_fn.clone().launch(cfg, (
                &self.d_logits,
                &self.output_layer.d_pre_activation,
                &fc_input_ptr,
                &self.output_layer.d_weights,
                &self.output_layer.d_biases,
                &self.output_layer.d_dropout_mask,
                &out_dims,
            ))?;
        }

        self.device.synchronize()?;

        let h_logits = self.device.dtoh_sync_copy(&self.d_logits)?;

        let max_val = h_logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = h_logits.iter().map(|x| (x - max_val).exp()).sum();

        let cfg = Self::launch_config(self.output_size as u32);
        unsafe {
            softmax_fn.launch(cfg, (
                &self.d_softmax_output,
                &self.d_logits,
                self.output_size,
                max_val,
                sum_exp,
            ))?;
        }

        self.device.synchronize()?;

        let result = self.device.dtoh_sync_copy(&self.d_softmax_output)?;
        self.device.dtod_copy(&self.d_softmax_output, &mut self.output_layer.d_output)?;

        Ok(result)
    }

    pub fn train_step(&mut self, image_data: &[f64], target: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
        self.is_training = true;

        let prediction = self.predict(image_data)?;

        self.device.htod_copy_into(target.to_vec(), &mut self.d_target)?;

        let output_grad: Vec<f64> = prediction.iter()
            .zip(target.iter())
            .map(|(p, t)| p - t)
            .collect();

        self.device.htod_copy_into(output_grad, &mut self.d_fc_grad)?;

        let fc_backward_fn = self.device.get_func("cnn_kernels", "fc_backward_kernel").unwrap();
        let fc_input_grad_fn = self.device.get_func("cnn_kernels", "fc_input_grad_kernel").unwrap();
        let pool_backward_fn = self.device.get_func("cnn_kernels", "pool_backward_kernel").unwrap();
        let apply_relu_deriv_fn = self.device.get_func("cnn_kernels", "apply_relu_deriv_kernel").unwrap();
        let conv_weight_grad_fn = self.device.get_func("cnn_kernels", "conv_weight_grad_kernel").unwrap();
        let conv_bias_grad_fn = self.device.get_func("cnn_kernels", "conv_bias_grad_kernel").unwrap();
        let adam_update_fn = self.device.get_func("cnn_kernels", "adam_update_kernel").unwrap();
        let zero_array_fn = self.device.get_func("cnn_kernels", "zero_array_kernel").unwrap();

        // Output layer backward
        let out_dims = self.device.htod_copy(vec![self.output_layer.num_neurons, self.output_layer.num_inputs, 1])?;
        let cfg = Self::launch_config(self.output_layer.num_neurons as u32);
        unsafe {
            fc_backward_fn.clone().launch(cfg, (
                &self.output_layer.d_errors,
                &self.d_fc_grad,
                &self.output_layer.d_weights,
                &self.output_layer.d_pre_activation,
                &self.output_layer.d_dropout_mask,
                &out_dims,
            ))?;
        }

        let prev_size = if self.fc_layers.is_empty() {
            self.flattened_size
        } else {
            self.fc_layers.last().unwrap().num_neurons
        };

        let mut d_fc_grad_temp = self.device.alloc_zeros::<f64>(prev_size as usize)?;
        let grad_dims = self.device.htod_copy(vec![self.output_layer.num_neurons, prev_size])?;
        let cfg = Self::launch_config(prev_size as u32);
        unsafe {
            fc_input_grad_fn.clone().launch(cfg, (
                &d_fc_grad_temp,
                &self.output_layer.d_errors,
                &self.output_layer.d_weights,
                &grad_dims,
            ))?;
        }

        // FC layers backward
        for i in (0..self.fc_layers.len()).rev() {
            let fc = &self.fc_layers[i];
            let fc_dims = self.device.htod_copy(vec![fc.num_neurons, fc.num_inputs, 0])?;
            let cfg = Self::launch_config(fc.num_neurons as u32);
            unsafe {
                fc_backward_fn.clone().launch(cfg, (
                    &fc.d_errors,
                    &d_fc_grad_temp,
                    &fc.d_weights,
                    &fc.d_pre_activation,
                    &fc.d_dropout_mask,
                    &fc_dims,
                ))?;
            }

            let prev_size = if i > 0 {
                self.fc_layers[i - 1].num_neurons
            } else {
                self.flattened_size
            };

            let new_grad = self.device.alloc_zeros::<f64>(prev_size as usize)?;
            let grad_dims = self.device.htod_copy(vec![fc.num_neurons, prev_size])?;
            let cfg = Self::launch_config(prev_size as u32);
            unsafe {
                fc_input_grad_fn.clone().launch(cfg, (
                    &new_grad,
                    &fc.d_errors,
                    &fc.d_weights,
                    &grad_dims,
                ))?;
            }
            d_fc_grad_temp = new_grad;
        }

        self.device.dtod_copy(&d_fc_grad_temp, &mut self.d_conv_grad)?;

        // Conv layers backward
        let mut current_grad = self.d_conv_grad.clone();
        let grad_c = self.last_conv_c;

        for i in (0..self.conv_layers.len()).rev() {
            if i < self.pool_layers.len() {
                let pool = &self.pool_layers[i];
                let conv = &self.conv_layers[i];
                let input_h = conv.output_h;
                let input_w = conv.output_w;

                let input_size = grad_c * input_h * input_w;
                let pool_input_grad = self.device.alloc_zeros::<f64>(input_size as usize)?;

                let cfg = Self::launch_config(input_size as u32);
                unsafe {
                    zero_array_fn.clone().launch(cfg, (&pool_input_grad, input_size))?;
                }

                let pool_dims = self.device.htod_copy(vec![grad_c, input_h, input_w, pool.output_h, pool.output_w, pool.pool_size])?;
                let output_size = grad_c * pool.output_h * pool.output_w;
                let cfg = Self::launch_config(output_size as u32);
                unsafe {
                    pool_backward_fn.clone().launch(cfg, (
                        &pool_input_grad,
                        &current_grad,
                        &pool.d_max_indices_y,
                        &pool.d_max_indices_x,
                        &pool_dims,
                    ))?;
                }
                current_grad = pool_input_grad;
            }

            let conv = &self.conv_layers[i];
            let output_size = conv.num_filters * conv.output_h * conv.output_w;
            let grad_with_relu = self.device.alloc_zeros::<f64>(output_size as usize)?;

            let cfg = Self::launch_config(output_size as u32);
            unsafe {
                apply_relu_deriv_fn.clone().launch(cfg, (
                    &grad_with_relu,
                    &current_grad,
                    &conv.d_pre_activation,
                    output_size,
                ))?;
            }

            let weight_size = conv.num_filters * conv.input_channels * conv.kernel_size * conv.kernel_size;
            let padded_h = if i > 0 {
                self.conv_layers[i - 1].output_h
            } else {
                self.input_height
            } + 2 * conv.padding;
            let padded_w = if i > 0 {
                self.conv_layers[i - 1].output_w
            } else {
                self.input_width
            } + 2 * conv.padding;

            let weight_dims = self.device.htod_copy(vec![
                conv.num_filters, conv.input_channels, conv.kernel_size,
                conv.output_h, conv.output_w, padded_h, padded_w, conv.stride
            ])?;
            let cfg = Self::launch_config(weight_size as u32);
            unsafe {
                conv_weight_grad_fn.clone().launch(cfg, (
                    &conv.d_weight_grads,
                    &grad_with_relu,
                    &conv.d_padded_input,
                    &weight_dims,
                ))?;
            }

            let bias_dims = self.device.htod_copy(vec![conv.num_filters, conv.output_h, conv.output_w])?;
            let cfg = Self::launch_config(conv.num_filters as u32);
            unsafe {
                conv_bias_grad_fn.clone().launch(cfg, (
                    &conv.d_bias_grads,
                    &grad_with_relu,
                    &bias_dims,
                ))?;
            }
        }

        self.device.synchronize()?;

        self.adam_t += 1;

        // Adam updates for conv layers
        let adam_params = self.device.htod_copy(vec![self.learning_rate, self.beta1, self.beta2])?;

        for conv in &self.conv_layers {
            let weight_size = conv.num_filters * conv.input_channels * conv.kernel_size * conv.kernel_size;
            let adam_dims = self.device.htod_copy(vec![weight_size, self.adam_t])?;
            let cfg = Self::launch_config(weight_size as u32);
            unsafe {
                adam_update_fn.clone().launch(cfg, (
                    &conv.d_weights,
                    &conv.d_weights_m,
                    &conv.d_weights_v,
                    &conv.d_weight_grads,
                    &adam_params,
                    &adam_dims,
                ))?;
            }

            let adam_dims = self.device.htod_copy(vec![conv.num_filters, self.adam_t])?;
            let cfg = Self::launch_config(conv.num_filters as u32);
            unsafe {
                adam_update_fn.clone().launch(cfg, (
                    &conv.d_biases,
                    &conv.d_bias_m,
                    &conv.d_bias_v,
                    &conv.d_bias_grads,
                    &adam_params,
                    &adam_dims,
                ))?;
            }
        }

        // Adam updates for FC layers
        for (idx, fc) in self.fc_layers.iter().enumerate() {
            let weight_size = fc.num_neurons * fc.num_inputs;

            let h_errors = self.device.dtoh_sync_copy(&fc.d_errors)?;
            let prev_out = if idx == 0 {
                self.device.dtoh_sync_copy(&self.d_flattened_features)?
            } else {
                self.device.dtoh_sync_copy(&self.fc_layers[idx - 1].d_output)?
            };

            let mut h_grads = vec![0.0f64; weight_size as usize];
            for i in 0..fc.num_neurons as usize {
                for j in 0..fc.num_inputs as usize {
                    h_grads[i * fc.num_inputs as usize + j] = h_errors[i] * prev_out[j];
                }
            }

            let d_fc_weight_grads = self.device.htod_copy(h_grads)?;

            let adam_dims = self.device.htod_copy(vec![weight_size, self.adam_t])?;
            let cfg = Self::launch_config(weight_size as u32);
            unsafe {
                adam_update_fn.clone().launch(cfg, (
                    &fc.d_weights,
                    &fc.d_weights_m,
                    &fc.d_weights_v,
                    &d_fc_weight_grads,
                    &adam_params,
                    &adam_dims,
                ))?;
            }

            let adam_dims = self.device.htod_copy(vec![fc.num_neurons, self.adam_t])?;
            let cfg = Self::launch_config(fc.num_neurons as u32);
            unsafe {
                adam_update_fn.clone().launch(cfg, (
                    &fc.d_biases,
                    &fc.d_bias_m,
                    &fc.d_bias_v,
                    &fc.d_errors,
                    &adam_params,
                    &adam_dims,
                ))?;
            }
        }

        // Output layer Adam update
        {
            let weight_size = self.output_layer.num_neurons * self.output_layer.num_inputs;

            let h_errors = self.device.dtoh_sync_copy(&self.output_layer.d_errors)?;
            let prev_out = if self.fc_layers.is_empty() {
                self.device.dtoh_sync_copy(&self.d_flattened_features)?
            } else {
                self.device.dtoh_sync_copy(&self.fc_layers.last().unwrap().d_output)?
            };

            let mut h_grads = vec![0.0f64; weight_size as usize];
            for i in 0..self.output_layer.num_neurons as usize {
                for j in 0..self.output_layer.num_inputs as usize {
                    h_grads[i * self.output_layer.num_inputs as usize + j] = h_errors[i] * prev_out[j];
                }
            }

            let d_out_weight_grads = self.device.htod_copy(h_grads)?;

            let adam_dims = self.device.htod_copy(vec![weight_size, self.adam_t])?;
            let cfg = Self::launch_config(weight_size as u32);
            unsafe {
                adam_update_fn.clone().launch(cfg, (
                    &self.output_layer.d_weights,
                    &self.output_layer.d_weights_m,
                    &self.output_layer.d_weights_v,
                    &d_out_weight_grads,
                    &adam_params,
                    &adam_dims,
                ))?;
            }

            let adam_dims = self.device.htod_copy(vec![self.output_layer.num_neurons, self.adam_t])?;
            let cfg = Self::launch_config(self.output_layer.num_neurons as u32);
            unsafe {
                adam_update_fn.clone().launch(cfg, (
                    &self.output_layer.d_biases,
                    &self.output_layer.d_bias_m,
                    &self.output_layer.d_bias_v,
                    &self.output_layer.d_errors,
                    &adam_params,
                    &adam_dims,
                ))?;
            }
        }

        self.device.synchronize()?;

        let loss = Loss::compute(&prediction, target, self.loss_function);
        Ok(loss)
    }

    pub fn save(&self, filename: &str) -> Result<bool, Box<dyn std::error::Error>> {
        let mut file = BufWriter::new(File::create(filename)?);

        file.write_all(MODEL_MAGIC)?;

        let write_i32 = |f: &mut BufWriter<File>, v: i32| -> std::io::Result<()> {
            f.write_all(&v.to_le_bytes())
        };
        let write_f64 = |f: &mut BufWriter<File>, v: f64| -> std::io::Result<()> {
            f.write_all(&v.to_le_bytes())
        };

        write_i32(&mut file, self.input_width)?;
        write_i32(&mut file, self.input_height)?;
        write_i32(&mut file, self.input_channels)?;
        write_i32(&mut file, self.output_size)?;
        write_f64(&mut file, self.learning_rate)?;
        write_f64(&mut file, self.dropout_rate)?;
        write_f64(&mut file, self.beta1)?;
        write_f64(&mut file, self.beta2)?;
        write_i32(&mut file, self.adam_t)?;
        write_i32(&mut file, self.flattened_size)?;
        write_i32(&mut file, self.last_conv_h)?;
        write_i32(&mut file, self.last_conv_w)?;
        write_i32(&mut file, self.last_conv_c)?;

        write_i32(&mut file, self.conv_layers.len() as i32)?;
        for &f in &self.f_conv_filters {
            write_i32(&mut file, f)?;
        }
        for &k in &self.f_kernel_sizes {
            write_i32(&mut file, k)?;
        }

        write_i32(&mut file, self.pool_layers.len() as i32)?;
        for &p in &self.f_pool_sizes {
            write_i32(&mut file, p)?;
        }

        write_i32(&mut file, self.fc_layers.len() as i32)?;
        for &fc in &self.f_fc_sizes {
            write_i32(&mut file, fc)?;
        }

        for conv in &self.conv_layers {
            let weights = self.device.dtoh_sync_copy(&conv.d_weights)?;
            let weights_m = self.device.dtoh_sync_copy(&conv.d_weights_m)?;
            let weights_v = self.device.dtoh_sync_copy(&conv.d_weights_v)?;
            let biases = self.device.dtoh_sync_copy(&conv.d_biases)?;
            let bias_m = self.device.dtoh_sync_copy(&conv.d_bias_m)?;
            let bias_v = self.device.dtoh_sync_copy(&conv.d_bias_v)?;

            for &w in &weights { write_f64(&mut file, w)?; }
            for &w in &weights_m { write_f64(&mut file, w)?; }
            for &w in &weights_v { write_f64(&mut file, w)?; }
            for &b in &biases { write_f64(&mut file, b)?; }
            for &b in &bias_m { write_f64(&mut file, b)?; }
            for &b in &bias_v { write_f64(&mut file, b)?; }
        }

        let save_fc_layer = |file: &mut BufWriter<File>, layer: &FCLayerGPU, device: &Arc<CudaDevice>| -> Result<(), Box<dyn std::error::Error>> {
            let weights = device.dtoh_sync_copy(&layer.d_weights)?;
            let weights_m = device.dtoh_sync_copy(&layer.d_weights_m)?;
            let weights_v = device.dtoh_sync_copy(&layer.d_weights_v)?;
            let biases = device.dtoh_sync_copy(&layer.d_biases)?;
            let bias_m = device.dtoh_sync_copy(&layer.d_bias_m)?;
            let bias_v = device.dtoh_sync_copy(&layer.d_bias_v)?;

            for &w in &weights { file.write_all(&w.to_le_bytes())?; }
            for &w in &weights_m { file.write_all(&w.to_le_bytes())?; }
            for &w in &weights_v { file.write_all(&w.to_le_bytes())?; }
            for &b in &biases { file.write_all(&b.to_le_bytes())?; }
            for &b in &bias_m { file.write_all(&b.to_le_bytes())?; }
            for &b in &bias_v { file.write_all(&b.to_le_bytes())?; }
            Ok(())
        };

        for fc in &self.fc_layers {
            save_fc_layer(&mut file, fc, &self.device)?;
        }
        save_fc_layer(&mut file, &self.output_layer, &self.device)?;

        Ok(true)
    }

    pub fn save_to_json(&self, filename: &str) -> Result<bool, Box<dyn std::error::Error>> {
        let mut file = File::create(filename)?;

        writeln!(file, "{{")?;
        writeln!(file, "  \"input_width\": {},", self.input_width)?;
        writeln!(file, "  \"input_height\": {},", self.input_height)?;
        writeln!(file, "  \"input_channels\": {},", self.input_channels)?;
        writeln!(file, "  \"output_size\": {},", self.output_size)?;
        writeln!(file, "  \"gradient_clip\": {:.6},", self.gradient_clip)?;

        write!(file, "  \"conv_filters\": [")?;
        for (i, f) in self.f_conv_filters.iter().enumerate() {
            if i > 0 { write!(file, ", ")?; }
            write!(file, "{}", f)?;
        }
        writeln!(file, "],")?;

        write!(file, "  \"kernel_sizes\": [")?;
        for (i, k) in self.f_kernel_sizes.iter().enumerate() {
            if i > 0 { write!(file, ", ")?; }
            write!(file, "{}", k)?;
        }
        writeln!(file, "],")?;

        write!(file, "  \"pool_sizes\": [")?;
        for (i, p) in self.f_pool_sizes.iter().enumerate() {
            if i > 0 { write!(file, ", ")?; }
            write!(file, "{}", p)?;
        }
        writeln!(file, "],")?;

        write!(file, "  \"fc_sizes\": [")?;
        for (i, fc) in self.f_fc_sizes.iter().enumerate() {
            if i > 0 { write!(file, ", ")?; }
            write!(file, "{}", fc)?;
        }
        writeln!(file, "],")?;

        writeln!(file, "  \"learning_rate\": {:.6},", self.learning_rate)?;
        writeln!(file, "  \"dropout_rate\": {:.6},", self.dropout_rate)?;
        writeln!(file, "  \"activation\": \"{}\",", self.hidden_activation.to_str())?;
        writeln!(file, "  \"output_activation\": \"{}\",", self.output_activation.to_str())?;
        writeln!(file, "  \"loss_type\": \"{}\",", self.loss_function.to_str())?;

        writeln!(file, "  \"conv_layers\": [")?;
        for (i, conv) in self.conv_layers.iter().enumerate() {
            let weights = self.device.dtoh_sync_copy(&conv.d_weights)?;
            let biases = self.device.dtoh_sync_copy(&conv.d_biases)?;

            writeln!(file, "    {{")?;
            writeln!(file, "      \"filters\": [")?;
            for f in 0..conv.num_filters {
                writeln!(file, "        {{")?;
                writeln!(file, "          \"bias\": {:.6},", biases[f as usize])?;
                write!(file, "          \"weights\": [")?;
                let filter_offset = (f * conv.input_channels * conv.kernel_size * conv.kernel_size) as usize;
                for c in 0..conv.input_channels {
                    if c > 0 { write!(file, ", ")?; }
                    write!(file, "[")?;
                    for ky in 0..conv.kernel_size {
                        if ky > 0 { write!(file, ", ")?; }
                        write!(file, "[")?;
                        for kx in 0..conv.kernel_size {
                            if kx > 0 { write!(file, ", ")?; }
                            let idx = filter_offset + (c * conv.kernel_size * conv.kernel_size + ky * conv.kernel_size + kx) as usize;
                            write!(file, "{:.6}", weights[idx])?;
                        }
                        write!(file, "]")?;
                    }
                    write!(file, "]")?;
                }
                writeln!(file, "]")?;
                if f < conv.num_filters - 1 {
                    writeln!(file, "        }},")?;
                } else {
                    writeln!(file, "        }}")?;
                }
            }
            writeln!(file, "      ]")?;
            if i < self.conv_layers.len() - 1 {
                writeln!(file, "    }},")?;
            } else {
                writeln!(file, "    }}")?;
            }
        }
        writeln!(file, "  ],")?;

        writeln!(file, "  \"pool_layers\": [")?;
        for (i, pool) in self.pool_layers.iter().enumerate() {
            write!(file, "    {{\"poolSize\": {}}}", pool.pool_size)?;
            if i < self.pool_layers.len() - 1 {
                writeln!(file, ",")?;
            } else {
                writeln!(file)?;
            }
        }
        writeln!(file, "  ],")?;

        writeln!(file, "  \"fc_layers\": [")?;
        for (i, fc) in self.fc_layers.iter().enumerate() {
            let weights = self.device.dtoh_sync_copy(&fc.d_weights)?;
            let biases = self.device.dtoh_sync_copy(&fc.d_biases)?;

            writeln!(file, "    {{")?;
            writeln!(file, "      \"neurons\": [")?;
            for j in 0..fc.num_neurons {
                writeln!(file, "        {{")?;
                writeln!(file, "          \"bias\": {:.6},", biases[j as usize])?;
                write!(file, "          \"weights\": [")?;
                for w in 0..fc.num_inputs {
                    if w > 0 { write!(file, ", ")?; }
                    write!(file, "{:.6}", weights[(j * fc.num_inputs + w) as usize])?;
                }
                writeln!(file, "]")?;
                if j < fc.num_neurons - 1 {
                    writeln!(file, "        }},")?;
                } else {
                    writeln!(file, "        }}")?;
                }
            }
            writeln!(file, "      ]")?;
            if i < self.fc_layers.len() - 1 {
                writeln!(file, "    }},")?;
            } else {
                writeln!(file, "    }}")?;
            }
        }
        writeln!(file, "  ],")?;

        writeln!(file, "  \"output_layer\": {{")?;
        writeln!(file, "    \"neurons\": [")?;
        {
            let weights = self.device.dtoh_sync_copy(&self.output_layer.d_weights)?;
            let biases = self.device.dtoh_sync_copy(&self.output_layer.d_biases)?;

            for j in 0..self.output_layer.num_neurons {
                writeln!(file, "      {{")?;
                writeln!(file, "        \"bias\": {:.6},", biases[j as usize])?;
                write!(file, "        \"weights\": [")?;
                for w in 0..self.output_layer.num_inputs {
                    if w > 0 { write!(file, ", ")?; }
                    write!(file, "{:.6}", weights[(j * self.output_layer.num_inputs + w) as usize])?;
                }
                writeln!(file, "]")?;
                if j < self.output_layer.num_neurons - 1 {
                    writeln!(file, "      }},")?;
                } else {
                    writeln!(file, "      }}")?;
                }
            }
        }
        writeln!(file, "    ]")?;
        writeln!(file, "  }}")?;
        writeln!(file, "}}")?;

        Ok(true)
    }

    pub fn get_input_width(&self) -> i32 { self.input_width }
    pub fn get_input_height(&self) -> i32 { self.input_height }
    pub fn get_input_channels(&self) -> i32 { self.input_channels }
    pub fn get_output_size(&self) -> i32 { self.output_size }
    pub fn get_num_conv_layers(&self) -> usize { self.conv_layers.len() }
    pub fn get_num_fc_layers(&self) -> usize { self.fc_layers.len() }
    pub fn get_learning_rate(&self) -> f64 { self.learning_rate }
    pub fn get_gradient_clip(&self) -> f64 { self.gradient_clip }
    pub fn get_hidden_activation(&self) -> ActivationType { self.hidden_activation }
    pub fn get_output_activation(&self) -> ActivationType { self.output_activation }
    pub fn get_loss_type(&self) -> LossType { self.loss_function }
    pub fn get_conv_filters(&self) -> &[i32] { &self.f_conv_filters }
    pub fn get_kernel_sizes(&self) -> &[i32] { &self.f_kernel_sizes }
    pub fn get_pool_sizes(&self) -> &[i32] { &self.f_pool_sizes }
    pub fn get_fc_sizes(&self) -> &[i32] { &self.f_fc_sizes }
    pub fn get_use_batch_norm(&self) -> bool { self.use_batch_norm }
    pub fn get_batch_norm_params(&self) -> &[BatchNormParams] { &self.batch_norm_params }

    pub fn initialize_batch_norm(&mut self) {
        self.use_batch_norm = true;
        self.batch_norm_params.clear();

        for conv in &self.conv_layers {
            let size = (conv.num_filters * conv.output_h * conv.output_w) as usize;
            let mut params = BatchNormParams::new();
            params.initialize(size);
            self.batch_norm_params.push(params);
        }
    }

    pub fn apply_batch_norm(&mut self, layer_idx: usize, data: &mut [f64], training: bool) {
        if !self.use_batch_norm || layer_idx >= self.batch_norm_params.len() {
            return;
        }

        let params = &mut self.batch_norm_params[layer_idx];
        let size = data.len();

        if training {
            let mean: f64 = data.iter().sum::<f64>() / size as f64;
            let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / size as f64;

            for i in 0..size {
                params.running_mean[i] = (1.0 - params.momentum) * params.running_mean[i] + params.momentum * mean;
                params.running_var[i] = (1.0 - params.momentum) * params.running_var[i] + params.momentum * var;
            }

            for i in 0..size {
                let normalized = (data[i] - mean) / (var + params.epsilon).sqrt();
                data[i] = params.gamma[i] * normalized + params.beta[i];
            }
        } else {
            for i in 0..size {
                let normalized = (data[i] - params.running_mean[i]) / (params.running_var[i] + params.epsilon).sqrt();
                data[i] = params.gamma[i] * normalized + params.beta[i];
            }
        }
    }

    pub fn export_to_onnx(&self, filename: &str) -> Result<bool, Box<dyn std::error::Error>> {
        use std::io::BufWriter;

        let mut file = BufWriter::new(File::create(filename)?);

        file.write_all(b"ONNX")?;
        file.write_all(&1u32.to_le_bytes())?;

        let write_i32 = |f: &mut BufWriter<File>, v: i32| -> std::io::Result<()> {
            f.write_all(&v.to_le_bytes())
        };
        let write_f64 = |f: &mut BufWriter<File>, v: f64| -> std::io::Result<()> {
            f.write_all(&v.to_le_bytes())
        };

        write_i32(&mut file, self.input_width)?;
        write_i32(&mut file, self.input_height)?;
        write_i32(&mut file, self.input_channels)?;
        write_i32(&mut file, self.output_size)?;
        write_f64(&mut file, self.learning_rate)?;
        write_f64(&mut file, self.gradient_clip)?;

        write_i32(&mut file, self.hidden_activation as i32)?;
        write_i32(&mut file, self.output_activation as i32)?;
        write_i32(&mut file, self.loss_function as i32)?;

        write_i32(&mut file, self.conv_layers.len() as i32)?;
        for &f in &self.f_conv_filters {
            write_i32(&mut file, f)?;
        }
        for &k in &self.f_kernel_sizes {
            write_i32(&mut file, k)?;
        }

        write_i32(&mut file, self.pool_layers.len() as i32)?;
        for &p in &self.f_pool_sizes {
            write_i32(&mut file, p)?;
        }

        write_i32(&mut file, self.fc_layers.len() as i32)?;
        for &fc in &self.f_fc_sizes {
            write_i32(&mut file, fc)?;
        }

        for conv in &self.conv_layers {
            let weights = self.device.dtoh_sync_copy(&conv.d_weights)?;
            let biases = self.device.dtoh_sync_copy(&conv.d_biases)?;

            for &w in &weights { write_f64(&mut file, w)?; }
            for &b in &biases { write_f64(&mut file, b)?; }
        }

        for fc in &self.fc_layers {
            let weights = self.device.dtoh_sync_copy(&fc.d_weights)?;
            let biases = self.device.dtoh_sync_copy(&fc.d_biases)?;

            for &w in &weights { write_f64(&mut file, w)?; }
            for &b in &biases { write_f64(&mut file, b)?; }
        }

        {
            let weights = self.device.dtoh_sync_copy(&self.output_layer.d_weights)?;
            let biases = self.device.dtoh_sync_copy(&self.output_layer.d_biases)?;

            for &w in &weights { write_f64(&mut file, w)?; }
            for &b in &biases { write_f64(&mut file, b)?; }
        }

        write_i32(&mut file, if self.use_batch_norm { 1 } else { 0 })?;
        if self.use_batch_norm {
            write_i32(&mut file, self.batch_norm_params.len() as i32)?;
            for params in &self.batch_norm_params {
                write_i32(&mut file, params.gamma.len() as i32)?;
                write_f64(&mut file, params.epsilon)?;
                write_f64(&mut file, params.momentum)?;
                for &g in &params.gamma { write_f64(&mut file, g)?; }
                for &b in &params.beta { write_f64(&mut file, b)?; }
                for &m in &params.running_mean { write_f64(&mut file, m)?; }
                for &v in &params.running_var { write_f64(&mut file, v)?; }
            }
        }

        Ok(true)
    }

    pub fn import_from_onnx(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        use std::io::{Read, BufReader};

        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"ONNX" {
            return Err("Invalid ONNX file magic".into());
        }

        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let _version = u32::from_le_bytes(version_bytes);

        let read_i32 = |r: &mut BufReader<File>| -> std::io::Result<i32> {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)?;
            Ok(i32::from_le_bytes(buf))
        };
        let read_f64 = |r: &mut BufReader<File>| -> std::io::Result<f64> {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
            Ok(f64::from_le_bytes(buf))
        };

        let input_width = read_i32(&mut reader)?;
        let input_height = read_i32(&mut reader)?;
        let input_channels = read_i32(&mut reader)?;
        let output_size = read_i32(&mut reader)?;
        let learning_rate = read_f64(&mut reader)?;
        let gradient_clip = read_f64(&mut reader)?;

        let hidden_act_i = read_i32(&mut reader)?;
        let output_act_i = read_i32(&mut reader)?;
        let loss_i = read_i32(&mut reader)?;

        let hidden_activation = match hidden_act_i {
            0 => ActivationType::Sigmoid,
            1 => ActivationType::Tanh,
            2 => ActivationType::ReLU,
            _ => ActivationType::Linear,
        };
        let output_activation = match output_act_i {
            0 => ActivationType::Sigmoid,
            1 => ActivationType::Tanh,
            2 => ActivationType::ReLU,
            _ => ActivationType::Linear,
        };
        let loss_function = match loss_i {
            1 => LossType::CrossEntropy,
            _ => LossType::MSE,
        };

        let num_conv = read_i32(&mut reader)? as usize;
        let mut conv_filters = Vec::with_capacity(num_conv);
        for _ in 0..num_conv {
            conv_filters.push(read_i32(&mut reader)?);
        }
        let mut kernel_sizes = Vec::with_capacity(num_conv);
        for _ in 0..num_conv {
            kernel_sizes.push(read_i32(&mut reader)?);
        }

        let num_pool = read_i32(&mut reader)? as usize;
        let mut pool_sizes = Vec::with_capacity(num_pool);
        for _ in 0..num_pool {
            pool_sizes.push(read_i32(&mut reader)?);
        }

        let num_fc = read_i32(&mut reader)? as usize;
        let mut fc_sizes = Vec::with_capacity(num_fc);
        for _ in 0..num_fc {
            fc_sizes.push(read_i32(&mut reader)?);
        }

        let mut cnn = Self::new(
            input_width, input_height, input_channels,
            &conv_filters, &kernel_sizes, &pool_sizes, &fc_sizes,
            output_size, hidden_activation, output_activation,
            loss_function, learning_rate, gradient_clip,
        )?;

        for conv in &mut cnn.conv_layers {
            let weight_size = (conv.num_filters * conv.input_channels * conv.kernel_size * conv.kernel_size) as usize;
            let mut weights = vec![0.0f64; weight_size];
            for w in &mut weights {
                *w = read_f64(&mut reader)?;
            }
            let mut biases = vec![0.0f64; conv.num_filters as usize];
            for b in &mut biases {
                *b = read_f64(&mut reader)?;
            }
            cnn.device.htod_copy_into(weights, &mut conv.d_weights)?;
            cnn.device.htod_copy_into(biases, &mut conv.d_biases)?;
        }

        for fc in &mut cnn.fc_layers {
            let weight_size = (fc.num_neurons * fc.num_inputs) as usize;
            let mut weights = vec![0.0f64; weight_size];
            for w in &mut weights {
                *w = read_f64(&mut reader)?;
            }
            let mut biases = vec![0.0f64; fc.num_neurons as usize];
            for b in &mut biases {
                *b = read_f64(&mut reader)?;
            }
            cnn.device.htod_copy_into(weights, &mut fc.d_weights)?;
            cnn.device.htod_copy_into(biases, &mut fc.d_biases)?;
        }

        {
            let weight_size = (cnn.output_layer.num_neurons * cnn.output_layer.num_inputs) as usize;
            let mut weights = vec![0.0f64; weight_size];
            for w in &mut weights {
                *w = read_f64(&mut reader)?;
            }
            let mut biases = vec![0.0f64; cnn.output_layer.num_neurons as usize];
            for b in &mut biases {
                *b = read_f64(&mut reader)?;
            }
            cnn.device.htod_copy_into(weights, &mut cnn.output_layer.d_weights)?;
            cnn.device.htod_copy_into(biases, &mut cnn.output_layer.d_biases)?;
        }

        let use_bn = read_i32(&mut reader)?;
        if use_bn == 1 {
            cnn.use_batch_norm = true;
            let num_bn = read_i32(&mut reader)? as usize;
            for _ in 0..num_bn {
                let size = read_i32(&mut reader)? as usize;
                let epsilon = read_f64(&mut reader)?;
                let momentum = read_f64(&mut reader)?;

                let mut gamma = vec![0.0f64; size];
                for g in &mut gamma { *g = read_f64(&mut reader)?; }
                let mut beta = vec![0.0f64; size];
                for b in &mut beta { *b = read_f64(&mut reader)?; }
                let mut running_mean = vec![0.0f64; size];
                for m in &mut running_mean { *m = read_f64(&mut reader)?; }
                let mut running_var = vec![0.0f64; size];
                for v in &mut running_var { *v = read_f64(&mut reader)?; }

                cnn.batch_norm_params.push(BatchNormParams {
                    gamma,
                    beta,
                    running_mean,
                    running_var,
                    epsilon,
                    momentum,
                });
            }
        }

        Ok(cnn)
    }
}
