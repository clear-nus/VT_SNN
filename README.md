# VT_SNN


## Prerequisites
1. [SLAYER](https://github.com/bamsumit/slayerPytorch) framework to learn Spiking Neural Network (SNN)
2. Install package requirements:
```
pip install -r requirements.txt
```
3. We use [guild.ai](https://github.com/guildai/guildai) to track
experiment runs.

# Pre-processing data

Preprocessing involves binning spikes, for both the visual and tactile
data. To run on beast:

```
    guild run vitac:preprocess-data
```

## Baseline Model

```
guild run vitac:baseline hidden_size=32 -b
```



