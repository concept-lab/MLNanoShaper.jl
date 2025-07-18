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
`
  --nb_epoch NB_EPOCH   the number of epochs to compute (type: Int64,
                        default: 0)
  --batch_size BATCH_SIZE
                        the size of the batch, must be configured in
                        function of VRAM size (type: Int64, default:
                        0)
  -m, --model MODEL     the model name (default: "")
  --van_der_waals_channel
                        whether to use van der Waals channel
  --smoothing           whether to enforce smoothing
  --nb_data_points NB_DATA_POINTS
                        the number of proteins in the dataset to use
                        (type: Int64, default: 0)
  -n, --name NAME       name of the training run (default: "")
  -c, --cutoff_radius CUTOFF_RADIUS
                        the cutoff_radius used in training (type:
                        Float32, default: 0.0)
  --ref_distance REF_DISTANCE
                        the reference distance (in A) used to rescale
                        distance to surface in loss (type: Float32,
                        default: 0.0)
  --loss LOSS           the loss function (default: "categorical")
  -l, --learning_rate LEARNING_RATE
                        the learning rate used by the model in
                        training (type: Float64, default: 1.0e-5)
  -g, --on_gpu          should we do the training on the gpu
  -h, --help            show this help message and exit
`
