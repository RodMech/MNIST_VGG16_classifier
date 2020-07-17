# Transfer learning: VGG16 (pretrained in Imagenet) to MNIST dataset


## Contents

This project is focused on how transfer learning can be useful for adapting an already trained VGG16 net (in Imagenet) to a classifier for the MNIST numbers dataset. 

The strategy has followed a canonical transfer learning pipeline, freezing the last layers and embedding into the net a new custom classifier. 

The net is able to work both in GPU and CPU. 



## Requirements

The required packages have been version-pinned in the `requirements.txt`.
The only required compatible dependencies are `docker` and `docker-compose` The following specifications have been tested (for experiment reproducibility):

`docker==19.03.8`

`docker-compose==1.25.0`

Install docker and docker-compose as suggested in the [official documentation](https://docs.docker.com/compose/install/).
Please, mind the compatibility of the versions, as suggested [here](https://docs.docker.com/compose/compose-file/). 

## Dockerfile for GPU and CPU

This version includes two Dockerfiles: 

- `dev-nogpu.dockerfile`

- `dev-gpu.dockerfile`: `CUDA 10 // CUDNN 7`

For training and testing inside the container, a docker-compose file is used. See `how to run it` below.


## Weights and training

The transfer learning pipeline has already been implemented by me. Find the weights in the following [link](https://drive.google.com/file/d/1VUaJSDC0C7ZzT_eAioyirjhxL1RALtkO/view?usp=sharing). 
Weights are expected to be placed in the project folder (`MNIST_VGG16_classifier/`).

The transfer learning training output details are the following: 

    Total epochs: 14. Best epoch: 11 with loss: 0.19 and acc: 93.55%
    8321.07 total seconds elapsed. 554.74 seconds per epoch.


## How to run it

There are two options available:

### Transfer learning stage

If the user wants to train and generate weights, or reproduce the results provided in this repository, you will need to run the code in a GPU. CPU shows very poor performance in this stage. 

The `dev-gpu.docker-compose.yml` file needs to be modified by the user, replacing the last line:

    command: bash -c "python3 transfer_learning.py"

instead of:

    command: bash -c "python3 test.py"

And execute: 

    docker-compose -f ./docker/dev-gpu.docker-compose.yml up

The weights will pop up in the project folder (`MNIST_VGG16_classifier/`), named as `MNIST_VGG16_transfer.pt`.

The expected required time is 90 minutes approximately. 

### Test 

Once the weights file has been generated (or downloaded), It is time to evaluate the model already trained with transfer learning.

There are two options, CPU inference and GPU inference. You should use consistently the target docker-compose `*.yml` file.
Make sure that the last line of the `dev-gpu.docker-compose.yml` or `dev-nogpu.docker-compose.yml` files has the following content: 

    command: bash -c "python3 test.py"

instead of:

    command: bash -c "python3 transfer_learning.py"

And execute: 

    docker-compose -f ./docker/<CPU/GPU-file>.yml up

The batch size has been adapted to GPU (64 items) and to CPU (16 items).

The console will output an overall accuracy, and a subsample of images is placed in the folder `./images/`.

The name of each image is an array of five components. Each component represents the choices that the network would have made if a classification decission shall be undertaken. 
For example, `[1 2 3 4 5].png` means that `1` is the first choice, `2` is the second and so on and so forth.

## Performance

- Inference (images per second)

| CPU (2,2 GHz Intel Core i7):  | GPU (Tesla T4) |
| ------------- | ------------- |
| 3 Hz  | 22 kHz  |

- Inference (seconds per epoch)

| GPU (Tesla T4) |
| ------------- |
| 475 |

## Citation

    @article{VGG16,
      title={Very Deep Convolutional Networks for Large-Scale Image Recognition},
      author={Simonyan, Karen and Zisserman, Andrew},
      journal={arXiv preprint arXiv:1409.1556},
      year={2014}
    }
    
    @article{lecun-mnisthandwrittendigit-2010,
      added-at = {2010-06-28T21:16:30.000+0200},
      author = {LeCun, Yann and Cortes, Corinna},
      howpublished = {http://yann.lecun.com/exdb/mnist/},
      keywords = {MSc _checked character_recognition mnist network neural},
      title = {{MNIST} handwritten digit database},
      url = {http://yann.lecun.com/exdb/mnist/},
      year = {2010}
    }
