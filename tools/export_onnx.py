# encoding: utf-8
"""
@modified: thanh
@contact: nguyenvanthanhhust@gmail.com
"""
import argparse
import os
import sys
from os import mkdir

import torch

sys.path.append('.')
from srcs.config import cfg
from srcs.modeling import build_model
from srcs.utils.logger import setup_logger

import onnxruntime as ort
import numpy as np

def export_pytorch_to_onnx(model, dummy_input, onnx_path, input_names=None, output_names=None, dynamic_axes=None):
    """
    Exports a PyTorch model to ONNX format.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        dummy_input (torch.Tensor or tuple of torch.Tensor): An example input to the model.
                                                              This is used to trace the model's computation graph.
        onnx_path (str): The path where the ONNX model will be saved.
        input_names (list of str, optional): Names for the input tensor(s).
        output_names (list of str, optional): Names for the output tensor(s).
        dynamic_axes (dict, optional): Specifies dynamic axes for inputs and outputs.
                                      Example: {'input1': {0: 'batch_size', 2: 'width'}, 'output1': {0: 'batch_size'}}
    """
    try:
        # Set the model to evaluation mode
        model.eval()

        # Export the model
        torch.onnx.export(model,
                          dummy_input,
                          onnx_path,
                          export_params=True,  # Store the trained parameter weights inside the model file
                          opset_version=13,    # The ONNX opset version to use
                          do_constant_folding=True, # Whether to execute constant folding for optimization
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print(f"Model successfully exported to ONNX at: {onnx_path}")

    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")

def main():
    parser = argparse.ArgumentParser(description="PyTorch Template Mini Imagenet Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("convert_onnx", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT))

    # Create a dummy input tensor
    # The shape of the dummy_input should match the expected input shape of your model
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    example_inputs = (torch.randn(1, 3, 224, 224),)
    onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)
    onnx_filename = str(cfg.TEST.WEIGHT).replace(".pth", ".onnx")
    onnx_program.save(onnx_filename)
    onnx_path = onnx_filename # Save in current directory for simplicity
    # # Define the output path for the ONNX model


    # # Define input and output names (optional but recommended for clarity)
    # input_names = ["input"]
    # # output_names = ["output"]

    # # Define dynamic axes (optional, useful if your model handles variable batch sizes or other dimensions)
    # # Here, we make the batch size dynamic for both input and output
    # # dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}


    # # Export the model
    # print("--- Exporting SimpleModel to ONNX ---")
    # # export_pytorch_to_onnx(model, dummy_input, onnx_path, input_names, output_names, dynamic_axes)
    # export_pytorch_to_onnx(model, dummy_input, onnx_path, input_names, output_names)

    # Load the ONNX model
    ort_session = ort.InferenceSession(onnx_path)

    # Get input and output names from the ONNX model
    # This is crucial as ONNX Runtime requires these names
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # Convert the dummy input tensor to NumPy
    # ONNX Runtime typically expects NumPy arrays
    onnx_input_np = dummy_input.cpu().numpy()

    # Run inference with ONNX Runtime
    # The run method returns a list of outputs
    ort_outputs = ort_session.run([output_name], {input_name: onnx_input_np})

    # Extract the output(s) from the list
    onnx_output_np = ort_outputs[0]
    print("ONNX Runtime Output Shape:", onnx_output_np.shape)
    print("ONNX Runtime Output (first 5 values):", onnx_output_np.flatten()[:5])

    with torch.no_grad(): # No need to calculate gradients for inference
        pytorch_output = model(dummy_input)
    # Convert PyTorch output to a NumPy array for comparison
    pytorch_output_np = pytorch_output.cpu().numpy()

    tolerance_rtol = 1e-05
    tolerance_atol = 1e-08

    are_outputs_close = np.allclose(pytorch_output_np, onnx_output_np, rtol=tolerance_rtol, atol=tolerance_atol)

    print(f"\nAre outputs numerically close? {are_outputs_close}")

    if not are_outputs_close:
        print("\nOutputs are NOT numerically close. Investigating differences:")
        diff = np.abs(pytorch_output_np - onnx_output_np)
        max_diff = np.max(diff)
        print(f"Maximum absolute difference: {max_diff}")

        # Optionally, print where the largest differences occur
        # Example for a single output tensor
        if diff.size > 0:
            max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"Index of max difference: {max_diff_idx}")
            print(f"PyTorch value at max diff: {pytorch_output_np[max_diff_idx]}")
            print(f"ONNX value at max diff: {onnx_output_np[max_diff_idx]}")

        # If differences are small but `allclose` fails, you might need to relax tolerances
        print(f"Consider adjusting rtol ({tolerance_rtol}) and atol ({tolerance_atol}) if differences are negligible.")

if __name__ == '__main__':
    main()
