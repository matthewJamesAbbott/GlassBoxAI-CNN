mod types;
mod kernels;
mod layers;
mod network;

pub use types::{ActivationType, LossType, BatchNormParams};
pub use network::ConvolutionalNeuralNetworkCUDA;
