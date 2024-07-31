# Cli Interface
The cli Interface can be accessed globaly after running `julia --project build/build.jl install`.

```
	~/.julia/bin/mlnanoshaper train [options] [flags]
```
Train a model.

## Intro

Train a model that can reconstruct a protein surface using Machine Learning.
Default value of parameters are specified in the `param/param.toml` file.
In order to override the param, you can use the differents options. 

## Options

- `--nb-epoch <Int>`: the number of epoch to compute.
- `--model, -m <String>`: the model name. Can be anakin.
- `--nb-data-points <Int>`: the number of proteins in the dataset to use
- `--name, -n <String>`: name of the training run
- `--cutoff-radius, -c <Float32>`: the cutoff_radius used in training
- `--ref-distance <Float32>`: the reference distane (in A) used to rescale distance to surface in loss
- `--learning-rate, -l <Float64>`: the learning rate use by the model in training.
- `--loss <String>`: the loss function, one of "categorical" or "continuous".

## Flags

- `--gpu, -g `: should we do the training on the gpu, does nothing currently.
