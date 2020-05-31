# Event-Driven Visual-Tactile Sensing and Learning for Robots

This work contributes an event-driven visual-tactile perception system, comprising a novel biologically-inspired tactile
sensor and multi-modal spike-based learning. Our biologically-inspired fingertip tactile sensor, NeuTouch, scales well with the
number of taxels thanks to its event-based nature. Likewise, our Visual-Tactile Spiking Neural Network (VT-SNN) enables fast
perception when coupled with event sensors. We evaluate our visual-tactile system (using the NeuTouch and Prophesee event camera) on two robot tasks: container classification and rotational slip detection.

<p align="center">
<img src="https://github.com/tasbolat1/VT_SNN/blob/master/auxiliary_files/VT_SNN.png" height="360" width="400">
</p>

## update
CNN3D models are being tested ...

## Overall structure of the codes
This git consists of two main parts: object classification and slip detection. For each task, we developed SNN model and its ANN counterpart. Please follow below instruction to run the code.

## Prerequisits

1. [SLAYER](https://github.com/bamsumit/slayerPytorch) framework to learn a Spiking Neural Network (SNN)
2. Install package requirements:
```
pip install -r requirements.txt
```
3. We use [guild.ai](https://github.com/guildai/guildai) to track
experiment runs. See training progress with `guild tensorboard`, and a quick overview
of metrics with `guild compare`.

## Pre-processing data

Preprocessing involves binning spikes, for both the visual and tactile
data.

1. 
```
    guild run vtsnn:prepare-data

```
2. 
```
    guild run vtsnn:downsample-vision

```
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
