# Event-Driven Visual-Tactile Sensing and Learning for Robots

This work contributes an event-driven visual-tactile perception system, comprising a novel biologically-inspired tactile
sensor and multi-modal spike-based learning. Our biologically-inspired fingertip tactile sensor, NeuTouch, scales well with the
number of taxels thanks to its event-based nature. Likewise, our Visual-Tactile Spiking Neural Network (VT-SNN) enables fast
perception when coupled with event sensors. We evaluate our visual-tactile system (using the NeuTouch and Prophesee event camera) on two robot tasks: container classification and rotational slip detection.


![architecture_VT_SNN](https://github.com/tasbolat1/VT_SNN/blob/master/auxillary_files/VT_SNN.pdf)

## update
CNN3D models are being tested ...

## Overall structure of the codes
This git consists of two main parts: object classification and slip detection. For each task, we developed SNN model and its ANN counterpart. Please follow below instruction to run the code.

## Prerequisits

1. [SLAYER](https://github.com/bamsumit/slayerPytorch) framework to learn Spiky Neural Network (SNN)
2. Install package requirements:
```
pip install -r requirements.txt
```
3. We use [guild.ai](https://github.com/guildai/guildai) to track
experiment runs. See training progress with `guild tensorboard`, and a quick overview
of metrics with `guild compare`.

## Pre-processing data

Preprocessing involves binning spikes, for both the visual and tactile
data. To run on beast:

1. 
```
    guild run vtsnn:prepare-data

```
2. 
```
    python downsample_images.py --path ../data_VT_SNN --count 400 --network network.yaml

```
## Models

1. Tactile unimodal

SNN:

```
guild run vtsnn:tact sample_file=1 -b
```
ANN:

```
guild run vtsnn:tact_ann sample_file=1 -b
```
2. Vision unimodal

SNN:

```
guild run vtsnn:viz sample_file=1 -b
```
ANN:

```
guild run vtsnn:viz_ann sample_file=1 -b
```
3. Multimodal (tactile + vision)

SNN:

```
guild run vtsnn:mm sample_file=1 -b
```
ANN:

```
guild run vtsnn:mm_ann sample_file=1 -b
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