use clap::{Parser, Subcommand};
use std::fs::File;
use std::io::{Read, BufReader};

use crate::cnn::{ActivationType, LossType, ConvolutionalNeuralNetworkCUDA};

#[derive(Parser)]
#[command(name = "cnn_cuda")]
#[command(about = "GlassBoxAI CNN - CUDA implementation using cudarc", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    Create {
        #[arg(long)]
        input_w: i32,
        #[arg(long)]
        input_h: i32,
        #[arg(long)]
        input_c: i32,
        #[arg(long)]
        conv: String,
        #[arg(long)]
        kernels: String,
        #[arg(long)]
        pools: String,
        #[arg(long)]
        fc: String,
        #[arg(long)]
        output: i32,
        #[arg(long)]
        save: String,
        #[arg(long, default_value = "0.001")]
        lr: f64,
        #[arg(long, default_value = "relu")]
        hidden_act: String,
        #[arg(long, default_value = "linear")]
        output_act: String,
        #[arg(long, default_value = "mse")]
        loss: String,
        #[arg(long, default_value = "5.0")]
        clip: f64,
    },
    Train {
        #[arg(long)]
        model: String,
        #[arg(long)]
        data: String,
        #[arg(long)]
        epochs: i32,
        #[arg(long)]
        save: String,
        #[arg(long, default_value = "32")]
        batch_size: i32,
    },
    Predict {
        #[arg(long)]
        model: String,
        #[arg(long)]
        data: String,
        #[arg(long)]
        output: String,
    },
    Info {
        #[arg(long)]
        model: String,
    },
    ExportOnnx {
        #[arg(long)]
        model: String,
        #[arg(long)]
        onnx: String,
    },
    ImportOnnx {
        #[arg(long)]
        onnx: String,
        #[arg(long)]
        save: String,
    },
    #[command(name = "usage")]
    Usage,
}

fn parse_int_list(s: &str) -> Vec<i32> {
    s.split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect()
}

pub fn print_help() {
    println!("Commands:");
    println!("  create   Create a new CNN model and save to JSON");
    println!("  train    Train an existing model with data from JSON");
    println!("  predict  Make predictions with a trained model from JSON");
    println!("  info     Display model information from JSON");
    println!("  export-onnx  Export model to ONNX binary format");
    println!("  import-onnx  Import model from ONNX binary format");
    println!("  usage    Show this help message\n");
    println!("Create Options:");
    println!("  --input-w=N            Input width (required)");
    println!("  --input-h=N            Input height (required)");
    println!("  --input-c=N            Input channels (required)");
    println!("  --conv=N,N,...         Conv filters (required)");
    println!("  --kernels=N,N,...      Kernel sizes (required)");
    println!("  --pools=N,N,...        Pool sizes (required)");
    println!("  --fc=N,N,...           FC layer sizes (required)");
    println!("  --output=N             Output layer size (required)");
    println!("  --save=FILE.json       Save model to JSON file (required)");
    println!("  --lr=VALUE             Learning rate (default: 0.001)");
    println!("  --hidden-act=TYPE      sigmoid|tanh|relu|linear (default: relu)");
    println!("  --output-act=TYPE      sigmoid|tanh|relu|linear (default: linear)");
    println!("  --loss=TYPE            mse|crossentropy (default: mse)");
    println!("  --clip=VALUE           Gradient clipping (default: 5.0)\n");
    println!("Train Options:");
    println!("  --model=FILE.json      Load model from JSON file (required)");
    println!("  --data=FILE.csv        Training data CSV file (required)");
    println!("  --epochs=N             Number of epochs (required)");
    println!("  --save=FILE.json       Save trained model to JSON (required)");
    println!("  --batch-size=N         Batch size (default: 32)\n");
    println!("Predict Options:");
    println!("  --model=FILE.json      Load model from JSON file (required)");
    println!("  --data=FILE.csv        Input data CSV file (required)");
    println!("  --output=FILE.csv      Save predictions to CSV file (required)\n");
    println!("Info Options:");
    println!("  --model=FILE.json      Load model from JSON file (required)\n");
    println!("Export ONNX Options:");
    println!("  --model=FILE.json      Load model from JSON file (required)");
    println!("  --onnx=FILE.onnx       Save to ONNX file (required)\n");
    println!("Import ONNX Options:");
    println!("  --onnx=FILE.onnx       Load from ONNX file (required)");
    println!("  --save=FILE.json       Save model to JSON file (required)\n");
    println!("Examples:");
    println!("  cnn_cuda create --input-w=28 --input-h=28 --input-c=1 --conv=32,64 --kernels=3,3 --pools=2,2 --fc=128 --output=10 --save=model.json");
    println!("  cnn_cuda train --model=model.json --data=data.csv --epochs=50 --save=model_trained.json");
    println!("  cnn_cuda predict --model=model_trained.json --data=test.csv --output=predictions.csv");
    println!("  cnn_cuda info --model=model.json");
    println!("  cnn_cuda export-onnx --model=model.json --onnx=model.onnx");
    println!("  cnn_cuda import-onnx --onnx=model.onnx --save=model.json");
}

pub fn print_model_info(model_file: &str) {
    let file = match File::open(model_file) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("Error: Cannot open model file: {}", model_file);
            return;
        }
    };

    let mut content = String::new();
    let mut reader = BufReader::new(file);
    if reader.read_to_string(&mut content).is_err() {
        eprintln!("Error: Cannot read model file");
        return;
    }

    let find_value = |key: &str| -> String {
        let search_key = format!("\"{}\": ", key);
        if let Some(pos) = content.find(&search_key) {
            let start = pos + search_key.len();
            let rest = &content[start..];
            let end = rest.find(|c| c == ',' || c == '\n' || c == '}').unwrap_or(rest.len());
            rest[..end].trim().to_string()
        } else {
            String::new()
        }
    };

    println!("\n=================================================================");
    println!("  Model Information:  {}", model_file);
    println!("=================================================================\n");
    println!("Architecture:");
    println!("Input: {}x{}x{}", find_value("input_width"), find_value("input_height"), find_value("input_channels"));
    println!("Output size: {}\n", find_value("output_size"));
    println!("Training Parameters:");
    println!("Learning rate: {}", find_value("learning_rate"));
    println!("Gradient clip: {}", find_value("gradient_clip"));
    println!("activation: {}", find_value("activation"));
    println!("output_activation: {}", find_value("output_activation"));
    println!("loss_type: {}\n", find_value("loss_type"));
}

pub fn handle_create(
    input_w: i32,
    input_h: i32,
    input_c: i32,
    conv: &str,
    kernels: &str,
    pools: &str,
    fc: &str,
    output: i32,
    save: &str,
    lr: f64,
    hidden_act: &str,
    output_act: &str,
    loss: &str,
    clip: f64,
) {
    let conv_filter_vec = parse_int_list(conv);
    let kernel_vec = parse_int_list(kernels);
    let pool_vec = parse_int_list(pools);
    let fc_vec = parse_int_list(fc);

    let hidden_act_type = ActivationType::from_str(hidden_act);
    let output_act_type = ActivationType::from_str(output_act);
    let loss_type = LossType::from_str(loss);

    println!("Creating CNN model...");
    println!("  Input: {}x{}x{}", input_w, input_h, input_c);
    println!("  Conv filters: {:?}", conv_filter_vec);
    println!("  Kernel sizes: {:?}", kernel_vec);
    println!("  Pool sizes: {:?}", pool_vec);
    println!("  FC layers: {:?}", fc_vec);
    println!("  Output size: {}", output);
    println!("  Hidden activation: {}", hidden_act_type.to_str());
    println!("  Output activation: {}", output_act_type.to_str());
    println!("  Loss function: {}", loss_type.to_str());
    println!("  Learning rate: {:.6}", lr);
    println!("  Gradient clip: {:.2}", clip);

    match ConvolutionalNeuralNetworkCUDA::new(
        input_w, input_h, input_c,
        &conv_filter_vec, &kernel_vec, &pool_vec, &fc_vec,
        output, hidden_act_type, output_act_type,
        loss_type, lr, clip,
    ) {
        Ok(cnn) => {
            match cnn.save_to_json(save) {
                Ok(_) => {
                    println!("Created CNN model");
                    println!("Model saved to: {}", save);
                }
                Err(e) => eprintln!("Error: Failed to save model: {}", e),
            }
        }
        Err(e) => eprintln!("Error: Failed to create CNN: {}", e),
    }
}

pub fn handle_train(model: &str, data: &str, epochs: i32, save: &str, batch_size: i32) {
    println!("Training model...");
    println!("  Model: {}", model);
    println!("  Data: {}", data);
    println!("  Epochs: {}", epochs);
    println!("  Batch size: {}", batch_size);
    println!("  Save to: {}\n", save);

    println!("Training not fully implemented in this CLI demo.");
    println!("To implement training:");
    println!("  1. Load CSV data from {}", data);
    println!("  2. Load model from {}", model);
    println!("  3. Run training loop with train_step() for {} epochs", epochs);
    println!("  4. Save updated model to {}", save);
    println!("\nSee the library API for complete training implementation.");
}

pub fn handle_predict(model: &str, data: &str, output: &str) {
    println!("Making predictions...");
    println!("  Model: {}", model);
    println!("  Data: {}", data);
    println!("  Output: {}\n", output);

    println!("Prediction not fully implemented in this CLI demo.");
    println!("To implement prediction:");
    println!("  1. Load model from {}", model);
    println!("  2. Load input data from CSV file: {}", data);
    println!("  3. Run predict() on each input");
    println!("  4. Save predictions to CSV: {}", output);
    println!("\nSee the library API for complete prediction implementation.");
}

pub fn handle_export_onnx(model: &str, onnx: &str) {
    println!("Exporting to ONNX...");
    println!("  Model: {}", model);
    println!("  ONNX: {}\n", onnx);

    println!("Export not fully implemented in this CLI demo.");
    println!("To implement export:");
    println!("  1. Load model from JSON file: {}", model);
    println!("  2. Call cnn.export_to_onnx(\"{}\")", onnx);
    println!("\nSee the library API for complete export implementation.");
}

pub fn handle_import_onnx(onnx: &str, save: &str) {
    println!("Importing from ONNX...");
    println!("  ONNX: {}", onnx);
    println!("  Save: {}\n", save);

    match ConvolutionalNeuralNetworkCUDA::import_from_onnx(onnx) {
        Ok(cnn) => {
            match cnn.save_to_json(save) {
                Ok(_) => {
                    println!("Imported model from ONNX");
                    println!("Model saved to: {}", save);
                }
                Err(e) => eprintln!("Error: Failed to save model: {}", e),
            }
        }
        Err(e) => eprintln!("Error: Failed to import ONNX: {}", e),
    }
}
