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

mod cnn;
mod cli;

use cli::{Cli, Commands};
use clap::Parser;

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Create {
            input_w, input_h, input_c, conv, kernels, pools, fc,
            output, save, lr, hidden_act, output_act, loss, clip,
        }) => {
            cli::handle_create(
                input_w, input_h, input_c, &conv, &kernels, &pools, &fc,
                output, &save, lr, &hidden_act, &output_act, &loss, clip,
            );
        }
        Some(Commands::Train { model, data, epochs, save, batch_size }) => {
            cli::handle_train(&model, &data, epochs, &save, batch_size);
        }
        Some(Commands::Predict { model, data, output }) => {
            cli::handle_predict(&model, &data, &output);
        }
        Some(Commands::Info { model }) => {
            cli::print_model_info(&model);
        }
        Some(Commands::ExportOnnx { model, onnx }) => {
            cli::handle_export_onnx(&model, &onnx);
        }
        Some(Commands::ImportOnnx { onnx, save }) => {
            cli::handle_import_onnx(&onnx, &save);
        }
        Some(Commands::Usage) | None => {
            cli::print_help();
        }
    }
}
