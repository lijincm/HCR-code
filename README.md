# HCR-code
# PyTorch implementation of "Hierarchical Correlations Replay for Continual Learning"

PyTorch implementation of "Hierarchical Correlations Replay for Continual Learning"

If you have any questions on this paper, feel free to create an issue.

## Acknowledgement

Our implementations use the source code from the following repository and user:

* [Dark Experience for General Continual Learning](https://github.com/aimagelab/mammoth)


# Meta-Transfer Learning PyTorch
[![Python](https://img.shields.io/badge/python-3.7-blue.svg?style=flat-square&logo=python&color=3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/torch-1.7.1-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)

#### Summary

* [Installation](#installation)
* [Project Architecture](#project-architecture)
* [Running Experiments](#running-experiments)
* [Hyperparameters](#Hyperparameters)


## Installation

In order to run this repository, we advise you to install python 3.7 and torch 1.7.1 and torchvision 0.8.0 with Anaconda.

You may download Anaconda and read the installation instruction on their official website:
<https://www.anaconda.com/download/>

Create a new environment and install PyTorch and torchvision on it:

```bash
conda create --name mtl-pytorch python=3.7
conda activate mtl-pytorch
conda install torch=1.7.1 
conda install torchvision=0.8.0
```
## Project Architecture

```
├── backbone
|   ├── untils
|   |   └── modules.py          # samplers for meta tra
|   └── ResNet18.py             # samplers for meta trai
├── datasets 
|   ├── untils
|   |   ├──continual_dataset.py # data loader for all datasets
|   |   └── validation.py       # samplers for meta trai
|   ├── transforms
|   |   └──denormalization.py   # data loader for all datasets
|   └── seq_cifar10.py          # samplers for meta train
├── models
|   ├── untils
|   |   └──continual_model.py   # data loader for all datasets
|   ├── hcr.py                  # meta-transfer class
|   ├── joint.py                # resnet class
|   └── sgd.py                  # meta-transfer convolution class
|   └── meta.py                 # meta-train trainer class
├── utils                       
|   ├── args.py                 # GPU tool functions
|   ├── best_args.py            # meta-transfer class
|   ├── buffer.py               # resnet class
|   ├── conf.py                 # GPU tool functions
|   ├── loggers.py              # meta-transfer class
|   ├── metrics.py              # resnet class
|   ├── status.py               # GPU tool functions
|   ├── tb_logger.py            # meta-transfer class
|   ├── training.py             # resnet class
|   └── triplet.py              # miscellaneous tool functions
├── main.py                     # the python file with main function and parameter settings
└── cifar.py                    # the script to run meta-train and meta-test phases
```

## Running Experiments
```
python mian.py --model xx --load_best_args --datasets xx --buffer_size xx 
or 
python cifar.py
```
## Hyperparameters
Hyperparameters in `main.py`.
```
'--dataset'         Which dataset to perform experiments on
'--model'           Model name
'--lr'              Learning rate
'--batch_size'      Batch size
'--n_epochs'        The number of epochs for each task
'--seed'            The random seed
'--csv_log'         Enable csv logging
'--buffer_size'     The size of the memory buffer
'--minibatch_size'  The batch size of the memory buffer
```
