@option struct ModelArgs
    van_der_waals_channel::Bool = false
end

"""
	train [options] [flags]

Train a model.

# Intro

Train a model that can reconstruct a protein surface using Machine Learning.
Default value of parameters are specified in the `param/param.toml` file.
In order to override the param, you can use the differents options. 

# Options

- `--nb-epoch <Int>`: the number of epoch to compute.
- `--model, -m <String>`: the model name. Can be anakin.
- `--nb-data-points <Int>`: the number of proteins in the dataset to use
- `--name, -n <String>`: name of the training run
- `--cutoff-radius, -c <Float32>`: the cutoff_radius used in training
- `--ref-distance <Float32>`: the reference distane (in A) used to rescale distance to surface in loss
- `--learning-rate, -l <Float64>`: the learning rate use by the model in training.
- `--loss <String>`: the loss function, one of "categorical" or "continuous".

# Flags

- `--gpu, -g `: should we do the training on the gpu, does nothing currently.

"""
@cast function train(; nb_epoch::Int = 0,
        model::String = "",
		model_kargs::ModelArgs=ModelArgs(),
        nb_data_points::Int = 0,
        name::String = "",
        cutoff_radius::Float32 = 0.0f0,
        ref_distance::Float32 = 00.0f0,
        loss::String,
        learning_rate::Float64 = 1e-5,
        gpu::Bool = false)
    global_logger(TerminalLogger())
    conf = TOML.parsefile(params_file)
    if nb_epoch > 0
        conf["AuxiliaryParameters"]["nb_epoch"] = nb_epoch |> UInt
    end
    if cutoff_radius != 0.0f0
        conf["TrainingParameters"]["cutoff_radius"] = cutoff_radius
    end
    if name != 0
        conf["TrainingParameters"]["name"] = name
    end
    if loss != 0
        conf["TrainingParameters"]["loss"] = loss
    end
    if learning_rate != 0.0
        conf["TrainingParameters"]["learning_rate"] = learning_rate
    end
    if ref_distance > 0
        conf["TrainingParameters"]["ref_distance"] = ref_distance
    end
    if model != ""
        conf["TrainingParameters"]["model"] = model
    end
    if model_kargs != ModelArgs()
        conf["TrainingParameters"]["model_kargs"] = Configurations.to_dict(model_kargs)
    end
    if nb_data_points > 0
        conf["TrainingParameters"]["data_ids"] = conf["TrainingParameters"]["data_ids"][begin:(begin + nb_data_points)]
    end

    training_parameters = read_from_TOML(TrainingParameters, conf)
    auxiliary_parameters = read_from_TOML(AuxiliaryParameters, conf)
    _train(training_parameters, auxiliary_parameters)
end

@main
