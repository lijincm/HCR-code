# HCR-code
# PyTorch implementation of "Hierarchical Correlations Replay for Continual Learning"

PyTorch implementation of "Hierarchical Correlations Replay for Continual Learning"

If you have any questions on this paper, feel free to create an issue.

## Acknowledgement

Our implementations use the source code from the following repository and user:

* [Dark Experience for General Continual Learning](https://github.com/aimagelab/mammoth)


# Hierarchical Correlations Replay for Continual Learning  PyTorch
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
|   ├── utils
|   |   └── modules.py          # 
|   └── ResNet18.py             # uses this net  extract features
├── datasets 
|   ├── utils
|   |   ├──continual_dataset.py # evaluation settings on dataset
|   |   └── validation.py       # validation dataset
|   ├── transforms
|   |   └──denormalization.py   # normalizes tensor
|   └── seq_cifar10.py          # data loader for cifar10
├── models
|   ├── utils
|   |   └──continual_model.py   # construct continual learning model
|   ├── hcr.py                  # our model
|   ├── joint.py                # upper bound
|   └── sgd.py                  # lower bound
├── utils                       
|   ├── args.py                 # adds the arguments used in the whole process
|   ├── best_args.py            # arguments
|   ├── buffer.py               # sampling methods and memory buffer
|   ├── conf.py                 # sets seeds and returns the device and base bath
|   ├── loggers.py              # logs the results
|   ├── metrics.py              # defines FWT and BWT
|   ├── status.py               # displays the process
|   ├── tb_logger.py            # logs current accuracy value and loss value
|   ├── training.py             # training process
|   └── triplet.py              # constructs triplets
├── main.py                     # the python file with main function 
└── cifar.py                    # the experiment on cifar-10
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
