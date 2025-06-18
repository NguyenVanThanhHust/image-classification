# Image classification
## Installation
Build docker image
```
docker build -t dev_img -f dockers/dev.Dockerfile ./dockers 
```
Build docker container
```
docker run -it --rm --name dev_ctn --gpus all --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/workspace -w /workspace dev_img /bin/bash
```
Minor note: check issue in Reference section

## Usage
Train model
```
python tools/train.py --config_file configs/train_mini_imnet.yml
```

Test model
```
python tools/test.py --config_file configs/train_mini_imnet.yml TEST.WEIGHT outputs/mini_imagenet/resnet_best.pth
```

Export to onnx format
```
python tools/export_onnx.py --config_file configs/train_mini_imnet.yml TEST.WEIGHT outputs/mini_imagenet/resnet_best.pth
```
The exported onnx model should be in the same folder as original pytorch model, in this case `outputs/mini_imagenet/resnet_best.onnx`

Export to tensorrt format
```
/usr/src/tensorrt/bin/trtexec \
            --onnx=outputs/mini_imagenet/resnet_best.onnx \
            --saveEngine=outputs/mini_imagenet/resnet_best.engine \
            --minShapes=input:1x3x224x224 \
            --optShapes=input:16x3x224x224 \
            --maxShapes=input:100x3x224x224
```
## Result

# Acknowledgments
This is copied and modified from [Deep-Learning-Project-Template](https://github.com/L1aoXingyu/Deep-Learning-Project-Template)

https://github.com/horovod/horovod/issues/2187#issuecomment-1238239742

