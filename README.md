# Event-Driven Visual-Tactile Sensing and Learning for Robots

This work contributes an event-driven visual-tactile perception system,
comprising a novel biologically-inspired tactile sensor and multi-modal
spike-based learning. Our biologically-inspired fingertip tactile sensor,
NeuTouch, scales well with the number of taxels thanks to its event-based
nature. Likewise, our Visual-Tactile Spiking Neural Network (VT-SNN) enables
fast perception when coupled with event sensors. We evaluate our visual-tactile
system (using the NeuTouch and Prophesee event camera) on two robot tasks:
container classification and rotational slip detection.

![img](img/VT_SNN.png)

## Requirements

The requirements for this project that can be installed from PyPI are found in
`requirements.txt`. To install the requirements, run:

``` bash
pip install -r requirements.txt
```

<!-- TODO -->

This project also requires a fork of the
[SLAYER](https://github.com/bamsumit/slayerPytorch) framework to learn a Spiking
Neural Network (SNN). To install this dependency, run:

``` bash
   git clone https://github.com/bamsumit/slayerPytorch
   cd slayerPytorch
   python setup.py install
```

This repository has been tested on Python 3.6.10.

## Usage

We provide the scripts for preprocessing the raw event data, and training the
models in the `vtsnn` folder. Instructions for running each script can be found
in each script. 

The repository has been carefully crafted to use
[guild.ai](https://github.com/guildai/guildai) to track experiment runs, and its
use is encouraged.

## Usage Instructions

Our scripts can be run using Guild or vanilla Python. To see all possible operations, run:

``` bash
guild operations
```

In this repository, we provide code for the 3 models presented in the paper:

1. VT-SNN (Using SLAYER)
2. ANN (MLP-GRU)
3. CNN3D

For example, to run our VT-SNN tactile-only model on the Container-Weight classification task, run:

``` bash
guild run vtsnn-snn:train-cw data_dir=/path/to/data
```

Visit the `vtsnn/train_*.py` files for instructions to run with vanilla Python.

## BibTeX

To cite this work, please use:

``` text
@inproceedings{taunyazov20event,
    title={Event-Driven Visual-Tactile Sensing and Learning for Robots}, 
    author={Tasbolat Taunyazoz and Weicong Sng and Hian Hian See and Brian Lim and Jethro Kuan and Abdul Fatir Ansari and Benjamin Tee and Harold Soh},
    year={2020},  
    booktitle = {Proceedings of Robotics: Science and Systems}, 
    year      = {2020}, 
    month     = {July}}
```
