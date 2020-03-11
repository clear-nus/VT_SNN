# VT_SNN

This repo contains all implementation given in the paper.

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