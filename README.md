# VT_SNN

## TODO
1. Run SNN models 5 times (TAS)
2. Where is the pooling layer code (JETHRO)
3. Modify ANNs (TAS and JETHRO)
more coming soon ...
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

```
    guild run vtsnn:prepare-data

```

## Baseline Model

```
guild run vitac:baseline hidden_size=32 -b
```



