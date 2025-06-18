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
## Usage
Train model
```
python tools/train.py --config_file configs/train_mnist_softmax.yml
```
## Result

# Acknowledgments
This is copied and modified from [Deep-Learning-Project-Template](https://github.com/L1aoXingyu/Deep-Learning-Project-Template)


