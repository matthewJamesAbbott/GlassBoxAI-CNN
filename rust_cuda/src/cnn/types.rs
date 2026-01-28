#![allow(dead_code)]

use serde::{Deserialize, Serialize};

pub const EPSILON: f64 = 1e-8;
pub const GRAD_CLIP: f64 = 1.0;
pub const BLOCK_SIZE: u32 = 256;
pub const MODEL_MAGIC: &[u8; 8] = b"CNNCUDA1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    Sigmoid,
    Tanh,
    ReLU,
    Linear,
}

impl ActivationType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "tanh" => ActivationType::Tanh,
            "relu" => ActivationType::ReLU,
            "linear" => ActivationType::Linear,
            _ => ActivationType::Sigmoid,
        }
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            ActivationType::Sigmoid => "sigmoid",
            ActivationType::Tanh => "tanh",
            ActivationType::ReLU => "relu",
            ActivationType::Linear => "linear",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossType {
    MSE,
    CrossEntropy,
}

impl LossType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "crossentropy" => LossType::CrossEntropy,
            _ => LossType::MSE,
        }
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            LossType::MSE => "mse",
            LossType::CrossEntropy => "crossentropy",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingType {
    Same,
    Valid,
}

pub struct Activation;

impl Activation {
    pub fn apply(x: f64, act_type: ActivationType) -> f64 {
        match act_type {
            ActivationType::Sigmoid => 1.0 / (1.0 + (-x.clamp(-500.0, 500.0)).exp()),
            ActivationType::Tanh => x.tanh(),
            ActivationType::ReLU => if x > 0.0 { x } else { 0.0 },
            ActivationType::Linear => x,
        }
    }

    pub fn derivative(y: f64, act_type: ActivationType) -> f64 {
        match act_type {
            ActivationType::Sigmoid => y * (1.0 - y),
            ActivationType::Tanh => 1.0 - y * y,
            ActivationType::ReLU => if y > 0.0 { 1.0 } else { 0.0 },
            ActivationType::Linear => 1.0,
        }
    }

    pub fn apply_softmax(arr: &mut [f64]) {
        let max_val = arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for v in arr.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        for v in arr.iter_mut() {
            *v /= sum;
        }
    }
}

pub struct Loss;

impl Loss {
    pub fn compute(pred: &[f64], target: &[f64], loss_type: LossType) -> f64 {
        let mut result = 0.0;
        match loss_type {
            LossType::MSE => {
                for (p, t) in pred.iter().zip(target.iter()) {
                    result += (p - t).powi(2);
                }
            }
            LossType::CrossEntropy => {
                for (p, t) in pred.iter().zip(target.iter()) {
                    let p_clamped = p.clamp(1e-15, 1.0 - 1e-15);
                    result -= t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln();
                }
            }
        }
        result / pred.len() as f64
    }

    pub fn gradient(pred: &[f64], target: &[f64], loss_type: LossType) -> Vec<f64> {
        let mut grad = vec![0.0; pred.len()];
        match loss_type {
            LossType::MSE => {
                for (i, (p, t)) in pred.iter().zip(target.iter()).enumerate() {
                    grad[i] = p - t;
                }
            }
            LossType::CrossEntropy => {
                for (i, (p, t)) in pred.iter().zip(target.iter()).enumerate() {
                    let p_clamped = p.clamp(1e-15, 1.0 - 1e-15);
                    grad[i] = (p_clamped - t) / (p_clamped * (1.0 - p_clamped) + 1e-15);
                }
            }
        }
        grad
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
