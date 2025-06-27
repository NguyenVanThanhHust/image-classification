import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

import matplotlib.pyplot as plt
from PIL import Image
import argparse

TRT_LOGGER = trt.Logger()

BATCH_SIZE = 1
INPUT_H = 224
INPUT_W = 224
INPUT_BLOB_NAME = "input"
OUTPUT_BLOB_NAME = "output"

ENGINE_PATH = "outputs/mini_imagenet/resnet_best.engine"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# For torchvision models, input images are loaded in to a range of [0, 1] and
# normalized using mean = [0.485, 0.456, 0.406] and stddev = [0.229, 0.224, 0.225].
def preprocess(image):
    # Mean normalization
    mean = np.array([0.5, 0.5, 0.5]).astype('float32')
    stddev = np.array([0.5, 0.5, 0.5]).astype('float32')
    data = (np.asarray(image).astype('float32') / float(255.0) - mean) / stddev
    # Switch from HWC to to CHW order
    return np.moveaxis(data, 2, 0)

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
    

def infer(engine, input_file):
    print("Reading input image from file {}".format(input_file))
    with Image.open(input_file) as img:
        input_image = preprocess(img)
        image_width = img.width
        image_height = img.height

    with engine.create_execution_context() as context:
        # Allocate host and device buffers
        tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        print(tensor_names)
        for tensor in tensor_names:
            size = trt.volume(context.get_tensor_shape(tensor))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor))
            print(size)
            print(context.get_tensor_shape(tensor))
            print(engine.get_tensor_mode(tensor))
            if engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT:
                context.set_input_shape(tensor, (1, 3, image_height, image_width))
                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_image.nbytes)
                context.set_tensor_address(tensor, int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                context.set_tensor_address(tensor, int(output_memory))

        stream = cuda.Stream()
        
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        
        # Run inference
        context.execute_async_v3(stream_handle=stream.handle)
        
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        
        # Synchronize the stream
        stream.synchronize()
        output_d64 = np.array(output_buffer, dtype=np.int64)
        print(type(output_d64))
        # np.savetxt('test.out', output_d64.astype(int), fmt='%i', delimiter=' ', newline=' ')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", action='store_true')
    parser.add_argument("-d", action='store_true')
    args = parser.parse_args()

    # if not (args.s ^ args.d):
    #     print(
    #         "arguments not right!\n"
    #         "python alexnet.py -d   # deserialize plan file and run inference"
    #     )
    #     sys.exit()

    runtime = trt.Runtime(TRT_LOGGER)
    assert runtime
    input_file = "tools/n01532829_110.JPEG"

    print("Running TensorRT inference for FCN-ResNet101")
    with load_engine(ENGINE_PATH) as engine:
        infer(engine, input_file)