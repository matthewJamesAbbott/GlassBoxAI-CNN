/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 */

use cxx_qt::CxxQtType;
use std::pin::Pin;

use crate::cnn::ConvolutionalNeuralNetworkCUDA;
use crate::cnn::{ActivationType, LossType};

#[cxx_qt::bridge]
pub mod qobject {
    unsafe extern "C++" {
        include!("cxx-qt-lib/qstring.h");
        type QString = cxx_qt_lib::QString;

        include!("cxx-qt-lib/qurl.h");
        type QUrl = cxx_qt_lib::QUrl;
    }

    unsafe extern "RustQt" {
        #[qobject]
        #[qml_element]
        #[qproperty(i32, training_progress)]
        #[qproperty(QString, training_status)]
        #[qproperty(QString, training_log)]
        #[qproperty(QString, metrics_text)]
        type CNNBridge = super::CNNBridgeRust;

        #[qsignal]
        fn training_finished(self: Pin<&mut CNNBridge>);

        #[qsignal]
        fn network_changed(self: Pin<&mut CNNBridge>);

        #[qinvokable]
        fn create_network(
            self: Pin<&mut CNNBridge>,
            width: i32,
            height: i32,
            channels: i32,
            conv_filters: &QString,
            kernel_sizes: &QString,
            pool_sizes: &QString,
            fc_sizes: &QString,
            output_size: i32,
            learning_rate: f64,
        ) -> QString;

        #[qinvokable]
        fn load_model(self: Pin<&mut CNNBridge>, url: &QUrl) -> QString;

        #[qinvokable]
        fn save_model(self: Pin<&mut CNNBridge>, url: &QUrl) -> QString;

        #[qinvokable]
        fn generate_synthetic_dataset(self: Pin<&mut CNNBridge>, samples_per_class: i32) -> QString;

        #[qinvokable]
        fn train_on_dataset(self: Pin<&mut CNNBridge>, epochs: i32, batch_size: i32);

        #[qinvokable]
        fn stop_training(self: Pin<&mut CNNBridge>);

        #[qinvokable]
        fn evaluate_model(self: Pin<&mut CNNBridge>) -> QString;

        #[qinvokable]
        fn predict_current_image(self: Pin<&mut CNNBridge>) -> QString;

        #[qinvokable]
        fn randomize_image(self: Pin<&mut CNNBridge>);

        #[qinvokable]
        fn get_feature_map(self: Pin<&mut CNNBridge>, layer_idx: i32, filter_idx: i32) -> QString;

        #[qinvokable]
        fn get_layer_config(self: Pin<&mut CNNBridge>, layer_idx: i32) -> QString;

        #[qinvokable]
        fn get_num_layers(self: Pin<&mut CNNBridge>) -> QString;

        #[qinvokable]
        fn get_num_conv_layers(self: Pin<&mut CNNBridge>) -> QString;

        #[qinvokable]
        fn get_num_fc_layers(self: Pin<&mut CNNBridge>) -> QString;

        #[qinvokable]
        fn get_optimizer_state(self: Pin<&mut CNNBridge>) -> QString;

        #[qinvokable]
        fn get_logits(self: Pin<&mut CNNBridge>) -> QString;

        #[qinvokable]
        fn get_softmax(self: Pin<&mut CNNBridge>) -> QString;

        #[qinvokable]
        fn set_feature_map_zeros(self: Pin<&mut CNNBridge>, layer_idx: i32, filter_idx: i32) -> QString;

        #[qinvokable]
        fn visualize_feature_map(self: Pin<&mut CNNBridge>, layer_idx: i32, filter_idx: i32) -> QString;

        #[qinvokable]
        fn get_pre_activation(self: Pin<&mut CNNBridge>, layer_idx: i32, filter_idx: i32) -> QString;

        #[qinvokable]
        fn set_pre_activation(self: Pin<&mut CNNBridge>, layer_idx: i32, filter_idx: i32, value: f64) -> QString;

        #[qinvokable]
        fn get_kernel(self: Pin<&mut CNNBridge>, layer_idx: i32, filter_idx: i32, channel_idx: i32) -> QString;

        #[qinvokable]
        fn set_kernel_random(self: Pin<&mut CNNBridge>, layer_idx: i32, filter_idx: i32, channel_idx: i32) -> QString;

        #[qinvokable]
        fn get_bias(self: Pin<&mut CNNBridge>, layer_idx: i32, filter_idx: i32) -> QString;

        #[qinvokable]
        fn set_bias(self: Pin<&mut CNNBridge>, layer_idx: i32, filter_idx: i32, value: f64) -> QString;

        #[qinvokable]
        fn visualize_all_kernels(self: Pin<&mut CNNBridge>, layer_idx: i32) -> QString;

        #[qinvokable]
        fn get_pooling_indices(self: Pin<&mut CNNBridge>, layer_idx: i32) -> QString;

        #[qinvokable]
        fn get_dropout_mask(self: Pin<&mut CNNBridge>, layer_idx: i32) -> QString;

        #[qinvokable]
        fn set_dropout_mask_ones(self: Pin<&mut CNNBridge>, layer_idx: i32) -> QString;

        #[qinvokable]
        fn get_filter_gradient(self: Pin<&mut CNNBridge>, layer_idx: i32, filter_idx: i32) -> QString;

        #[qinvokable]
        fn get_bias_gradient(self: Pin<&mut CNNBridge>, layer_idx: i32, filter_idx: i32) -> QString;

        #[qinvokable]
        fn get_flattened_features(self: Pin<&mut CNNBridge>) -> QString;

        #[qinvokable]
        fn get_num_filters(self: Pin<&mut CNNBridge>, layer_idx: i32) -> QString;

        #[qinvokable]
        fn get_layer_stats(self: Pin<&mut CNNBridge>, layer_idx: i32) -> QString;

        #[qinvokable]
        fn get_activation_histogram(self: Pin<&mut CNNBridge>, layer_idx: i32, num_bins: i32) -> QString;

        #[qinvokable]
        fn get_weight_histogram(self: Pin<&mut CNNBridge>, layer_idx: i32, num_bins: i32) -> QString;

        #[qinvokable]
        fn get_receptive_field(self: Pin<&mut CNNBridge>, layer_idx: i32, filter_idx: i32, y: i32, x: i32) -> QString;
    }
}

use cxx_qt_lib::QString;

fn parse_int_list(s: &str) -> Vec<i32> {
    s.split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect()
}

#[derive(Default)]
pub struct CNNBridgeRust {
    training_progress: i32,
    training_status: QString,
    training_log: QString,
    metrics_text: QString,

    cnn: Option<ConvolutionalNeuralNetworkCUDA>,
    current_image: Vec<f64>,
    dataset_inputs: Vec<Vec<f64>>,
    dataset_labels: Vec<Vec<f64>>,
    stop_requested: bool,
}

impl qobject::CNNBridge {
    pub fn create_network(
        mut self: Pin<&mut Self>,
        width: i32,
        height: i32,
        channels: i32,
        conv_filters: &QString,
        kernel_sizes: &QString,
        pool_sizes: &QString,
        fc_sizes: &QString,
        output_size: i32,
        learning_rate: f64,
    ) -> QString {
        let conv_filters_vec = parse_int_list(&conv_filters.to_string());
        let kernel_sizes_vec = parse_int_list(&kernel_sizes.to_string());
        let pool_sizes_vec = parse_int_list(&pool_sizes.to_string());
        let fc_sizes_vec = parse_int_list(&fc_sizes.to_string());

        match ConvolutionalNeuralNetworkCUDA::new(
            width,
            height,
            channels,
            &conv_filters_vec,
            &kernel_sizes_vec,
            &pool_sizes_vec,
            &fc_sizes_vec,
            output_size,
            ActivationType::ReLU,
            ActivationType::Linear,
            LossType::CrossEntropy,
            learning_rate,
            5.0,
        ) {
            Ok(cnn) => {
                self.as_mut().rust_mut().cnn = Some(cnn);
                self.as_mut().rust_mut().current_image = vec![0.0; (width * height * channels) as usize];
                self.network_changed();
                QString::from("Network created successfully")
            }
            Err(e) => QString::from(format!("Error: {}", e)),
        }
    }

    pub fn load_model(self: Pin<&mut Self>, url: &cxx_qt_lib::QUrl) -> QString {
        let path = url.to_string().replace("file://", "");
        QString::from(format!("Load model from {} - not yet implemented", path))
    }

    pub fn save_model(self: Pin<&mut Self>, url: &cxx_qt_lib::QUrl) -> QString {
        let path = url.to_string().replace("file://", "");
        if let Some(ref cnn) = self.cnn {
            match cnn.save_to_json(&path) {
                Ok(_) => QString::from(format!("Model saved to {}", path)),
                Err(e) => QString::from(format!("Error saving: {}", e)),
            }
        } else {
            QString::from("No network to save")
        }
    }

    pub fn generate_synthetic_dataset(mut self: Pin<&mut Self>, samples_per_class: i32) -> QString {
        let (width, height, channels, output_size) = {
            if let Some(ref cnn) = self.cnn {
                (cnn.get_input_width(), cnn.get_input_height(), cnn.get_input_channels(), cnn.get_output_size())
            } else {
                return QString::from("Create a network first");
            }
        };

        let image_size = (width * height * channels) as usize;

        self.as_mut().rust_mut().dataset_inputs.clear();
        self.as_mut().rust_mut().dataset_labels.clear();

        use rand::Rng;
        let mut rng = rand::thread_rng();

        for class_idx in 0..output_size {
            for _ in 0..samples_per_class {
                let mut image: Vec<f64> = (0..image_size).map(|_| rng.gen::<f64>() * 0.1).collect();

                let center_x = (width / 2) + (class_idx % 3 - 1) * (width / 6);
                let center_y = (height / 2) + (class_idx / 3 - 1) * (height / 6);
                let radius = (width.min(height) / 6) as f64;

                for y in 0..height {
                    for x in 0..width {
                        let dx = (x - center_x) as f64;
                        let dy = (y - center_y) as f64;
                        if dx * dx + dy * dy < radius * radius {
                            for c in 0..channels {
                                let idx = (c * height * width + y * width + x) as usize;
                                if idx < image_size {
                                    image[idx] = 0.8 + rng.gen::<f64>() * 0.2;
                                }
                            }
                        }
                    }
                }

                let mut label = vec![0.0; output_size as usize];
                label[class_idx as usize] = 1.0;

                self.as_mut().rust_mut().dataset_inputs.push(image);
                self.as_mut().rust_mut().dataset_labels.push(label);
            }
        }

        let total = self.dataset_inputs.len();
        QString::from(format!("Generated {} samples ({} classes × {} samples)", total, output_size, samples_per_class))
    }

    pub fn train_on_dataset(mut self: Pin<&mut Self>, epochs: i32, _batch_size: i32) {
        self.as_mut().rust_mut().stop_requested = false;
        self.as_mut().set_training_log(QString::from(""));

        let has_data = !self.dataset_inputs.is_empty();
        let has_cnn = self.cnn.is_some();

        if !has_cnn || !has_data {
            self.as_mut().set_training_status(QString::from("Create network and dataset first"));
            return;
        }

        let num_samples = self.dataset_inputs.len();

        for epoch in 0..epochs {
            if self.stop_requested {
                break;
            }

            let mut total_loss = 0.0;

            for i in 0..num_samples {
                if self.stop_requested {
                    break;
                }

                let input = self.dataset_inputs[i].clone();
                let target = self.dataset_labels[i].clone();

                let loss = if let Some(ref mut cnn) = self.as_mut().rust_mut().cnn {
                    cnn.train_step(&input, &target).unwrap_or(0.0)
                } else {
                    0.0
                };

                total_loss += loss;
            }

            let avg_loss = total_loss / num_samples as f64;
            let progress = ((epoch + 1) as f32 / epochs as f32 * 100.0) as i32;

            self.as_mut().set_training_progress(progress);

            let status = format!("Epoch {}/{} - Loss: {:.4}", epoch + 1, epochs, avg_loss);
            self.as_mut().set_training_status(QString::from(&status));

            let current_log = self.training_log().to_string();
            let new_log = format!("{}{}\n", current_log, status);
            self.as_mut().set_training_log(QString::from(&new_log));
        }

        self.training_finished();
    }

    pub fn stop_training(mut self: Pin<&mut Self>) {
        self.as_mut().rust_mut().stop_requested = true;
    }

    pub fn evaluate_model(mut self: Pin<&mut Self>) -> QString {
        if self.cnn.is_none() || self.dataset_inputs.is_empty() {
            return QString::from("Create network and dataset first");
        }

        let mut correct = 0;
        let total = self.dataset_inputs.len();

        for i in 0..total {
            let input = self.dataset_inputs[i].clone();
            let target = self.dataset_labels[i].clone();

            let prediction = if let Some(ref mut cnn) = self.as_mut().rust_mut().cnn {
                cnn.predict(&input).ok()
            } else {
                None
            };

            if let Some(pred) = prediction {
                let pred_class = pred.iter()
                    .enumerate()
                    .max_by(|(_, a): &(usize, &f64), (_, b): &(usize, &f64)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                let true_class = target.iter()
                    .enumerate()
                    .max_by(|(_, a): &(usize, &f64), (_, b): &(usize, &f64)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                if pred_class == true_class {
                    correct += 1;
                }
            }
        }

        let accuracy = correct as f64 / total as f64 * 100.0;
        QString::from(format!("Accuracy: {:.2}% ({}/{})", accuracy, correct, total))
    }

    pub fn predict_current_image(mut self: Pin<&mut Self>) -> QString {
        if self.cnn.is_none() {
            return QString::from("Create network first");
        }

        let input = self.current_image.clone();
        let prediction = if let Some(ref mut cnn) = self.as_mut().rust_mut().cnn {
            cnn.predict(&input).ok()
        } else {
            None
        };

        match prediction {
            Some(pred) => {
                let result: Vec<String> = pred.iter()
                    .enumerate()
                    .map(|(i, p)| format!("Class {}: {:.4}", i, p))
                    .collect();
                QString::from(result.join("\n"))
            }
            None => QString::from("Prediction failed"),
        }
    }

    pub fn randomize_image(mut self: Pin<&mut Self>) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for v in self.as_mut().rust_mut().current_image.iter_mut() {
            *v = rng.gen();
        }
    }

    pub fn get_feature_map(self: Pin<&mut Self>, layer_idx: i32, filter_idx: i32) -> QString {
        QString::from(format!("Feature map [layer={}, filter={}] - requires forward pass", layer_idx, filter_idx))
    }

    pub fn get_layer_config(self: Pin<&mut Self>, layer_idx: i32) -> QString {
        QString::from(format!("Layer {} configuration", layer_idx))
    }

    pub fn get_num_layers(self: Pin<&mut Self>) -> QString {
        if let Some(ref cnn) = self.cnn {
            let total = cnn.get_num_conv_layers() + cnn.get_num_fc_layers() + 1;
            QString::from(format!("Total layers: {}", total))
        } else {
            QString::from("No network")
        }
    }

    pub fn get_num_conv_layers(self: Pin<&mut Self>) -> QString {
        if let Some(ref cnn) = self.cnn {
            QString::from(format!("Conv layers: {}", cnn.get_num_conv_layers()))
        } else {
            QString::from("No network")
        }
    }

    pub fn get_num_fc_layers(self: Pin<&mut Self>) -> QString {
        if let Some(ref cnn) = self.cnn {
            QString::from(format!("FC layers: {}", cnn.get_num_fc_layers()))
        } else {
            QString::from("No network")
        }
    }

    pub fn get_optimizer_state(self: Pin<&mut Self>) -> QString {
        if let Some(ref cnn) = self.cnn {
            QString::from(format!("Adam optimizer - LR: {}", cnn.get_learning_rate()))
        } else {
            QString::from("No network")
        }
    }

    pub fn get_logits(self: Pin<&mut Self>) -> QString {
        QString::from("Logits (pre-softmax output)")
    }

    pub fn get_softmax(self: Pin<&mut Self>) -> QString {
        QString::from("Softmax output (class probabilities)")
    }

    pub fn set_feature_map_zeros(self: Pin<&mut Self>, layer_idx: i32, filter_idx: i32) -> QString {
        QString::from(format!("Set feature map [layer={}, filter={}] to zeros - not yet implemented", layer_idx, filter_idx))
    }

    pub fn visualize_feature_map(self: Pin<&mut Self>, layer_idx: i32, filter_idx: i32) -> QString {
        QString::from(format!("Visualize feature map [layer={}, filter={}] - not yet implemented", layer_idx, filter_idx))
    }

    pub fn get_pre_activation(self: Pin<&mut Self>, layer_idx: i32, filter_idx: i32) -> QString {
        QString::from(format!("Pre-activation [layer={}, filter={}] - not yet implemented", layer_idx, filter_idx))
    }

    pub fn set_pre_activation(self: Pin<&mut Self>, layer_idx: i32, filter_idx: i32, value: f64) -> QString {
        QString::from(format!("Set pre-activation [layer={}, filter={}] to {} - not yet implemented", layer_idx, filter_idx, value))
    }

    pub fn get_kernel(self: Pin<&mut Self>, layer_idx: i32, filter_idx: i32, channel_idx: i32) -> QString {
        QString::from(format!("Kernel [layer={}, filter={}, channel={}] - not yet implemented", layer_idx, filter_idx, channel_idx))
    }

    pub fn set_kernel_random(self: Pin<&mut Self>, layer_idx: i32, filter_idx: i32, channel_idx: i32) -> QString {
        QString::from(format!("Set kernel [layer={}, filter={}, channel={}] to random - not yet implemented", layer_idx, filter_idx, channel_idx))
    }

    pub fn get_bias(self: Pin<&mut Self>, layer_idx: i32, filter_idx: i32) -> QString {
        QString::from(format!("Bias [layer={}, filter={}] - not yet implemented", layer_idx, filter_idx))
    }

    pub fn set_bias(self: Pin<&mut Self>, layer_idx: i32, filter_idx: i32, value: f64) -> QString {
        QString::from(format!("Set bias [layer={}, filter={}] to {} - not yet implemented", layer_idx, filter_idx, value))
    }

    pub fn visualize_all_kernels(self: Pin<&mut Self>, layer_idx: i32) -> QString {
        QString::from(format!("Visualize all kernels [layer={}] - not yet implemented", layer_idx))
    }

    pub fn get_pooling_indices(self: Pin<&mut Self>, layer_idx: i32) -> QString {
        QString::from(format!("Pooling indices [layer={}] - not yet implemented", layer_idx))
    }

    pub fn get_dropout_mask(self: Pin<&mut Self>, layer_idx: i32) -> QString {
        QString::from(format!("Dropout mask [layer={}] - not yet implemented", layer_idx))
    }

    pub fn set_dropout_mask_ones(self: Pin<&mut Self>, layer_idx: i32) -> QString {
        QString::from(format!("Set dropout mask [layer={}] to ones - not yet implemented", layer_idx))
    }

    pub fn get_filter_gradient(self: Pin<&mut Self>, layer_idx: i32, filter_idx: i32) -> QString {
        QString::from(format!("Filter gradient [layer={}, filter={}] - not yet implemented", layer_idx, filter_idx))
    }

    pub fn get_bias_gradient(self: Pin<&mut Self>, layer_idx: i32, filter_idx: i32) -> QString {
        QString::from(format!("Bias gradient [layer={}, filter={}] - not yet implemented", layer_idx, filter_idx))
    }

    pub fn get_flattened_features(self: Pin<&mut Self>) -> QString {
        QString::from("Flattened features - not yet implemented")
    }

    pub fn get_num_filters(self: Pin<&mut Self>, layer_idx: i32) -> QString {
        QString::from(format!("Num filters [layer={}] - not yet implemented", layer_idx))
    }

    pub fn get_layer_stats(self: Pin<&mut Self>, layer_idx: i32) -> QString {
        QString::from(format!("Layer stats [layer={}] - not yet implemented", layer_idx))
    }

    pub fn get_activation_histogram(self: Pin<&mut Self>, layer_idx: i32, num_bins: i32) -> QString {
        QString::from(format!("Activation histogram [layer={}, bins={}] - not yet implemented", layer_idx, num_bins))
    }

    pub fn get_weight_histogram(self: Pin<&mut Self>, layer_idx: i32, num_bins: i32) -> QString {
        QString::from(format!("Weight histogram [layer={}, bins={}] - not yet implemented", layer_idx, num_bins))
    }

    pub fn get_receptive_field(self: Pin<&mut Self>, layer_idx: i32, filter_idx: i32, y: i32, x: i32) -> QString {
        QString::from(format!("Receptive field [layer={}, filter={}, y={}, x={}] - not yet implemented", layer_idx, filter_idx, y, x))
    }
}
