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

## Example End-to-end Workflow

### Data Processing
First, fetch the data:

<!-- TODO -->
``` bash
 wget ...
```
<!-- TODO: SLIP -->

Next, preprocess the data:

``` bash
    guild run preprocess 
```

In our experiments, we also downsample the image data using pooling, before
passing them as input to speed up training and reduce the input size. These are
required to match our model input dimensions. To run the downsampling, run:

``` bash
    guild run downsample
```

### Training the Models

#### SNN

#### ANN

#### 

## Models

1. Tactile unimodal

SNN:

```
guild run vtsnn:tact-snn sample_file=1 -b
```
ANN:

```
guild run vtsnn:tact-ann sample_file=1 -b
```
CNN3D:

```
guild run vtsnn:tact-cnn3d sample_file=1 -b
```
2. Vision unimodal

SNN:

```
guild run vtsnn:vis-snn sample_file=1 -b
```
ANN:

```
guild run vtsnn:vis-ann sample_file=1 -b
```
CNN3D:

```
guild run vtsnn:vis-cnn3d sample_file=1 -b
```
3. Multimodal (tactile + vision)

SNN:

```
guild run vtsnn:mm-snn sample_file=1 -b
```
ANN:

```
guild run vtsnn:mm-ann sample_file=1 -b
```
CNN3D:

```
guild run vtsnn:mm-cnn3d sample_file=1 -b
```
We have also implemented representation learning based on EST (RPG). Within ```rpg_event_representation_learning```, run

1. Tact ETS:
```
guild run vtsnn:tac_rpg sample_file=1 -b
```
2. Vision ETS:
```
guild run vtsnn:vis_rpg sample_file=1 -b
```
