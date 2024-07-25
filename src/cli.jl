using Comonicon
using Serialization
using Random
using Configurations
using Logging: global_logger
using TerminalLoggers: TerminalLogger
using LuxCUDA

@option struct ModelArgs
    van_der_waals_channel::Bool = false
end

"""
	train [options] [flags]

Train a model.

# Intro

Parameters are specified in the `param/param.toml` file.
The folowing parameters can be overided.

# Options

- `--nb-epoch, -e <Int>`: the number of epoch to compute.
- `--model, -m <String>`: the model name. Can be anakin.
- `--nb-data-points, -d <Int>`: the number of proteins in the dataset to use
- `--name, -n <String>`: name of the training run
- `--cutoff-radius, -c <Float32>`: the cutoff_radius used in training
- `--ref-distance, -d <Float32>`: the reference distane (in A) used to rescale distance to surface in loss

# Flags

- `--gpu, -g `: should we do the training on the gpu
- `--categorical, -c `: should we use the categorical version of the model

"""
@cast function train(; nb_epoch::Int = 0,
        model::String = "",
        model_kargs::ModelArgs,
        nb_data_points::Int = 0,
        name::String = "",
        cutoff_radius::Float32 = 0.0f0,
        ref_distance::Float32 = 00.0f0,
        categorical::Bool,
        learning_rate::Float64=0.,
        gpu::Bool = false)
    global_logger(TerminalLogger())
    conf = TOML.parsefile(params_file)
    conf["TrainingParameters"]["categorical"] = categorical
    if nb_epoch > 0
        conf["AuxiliaryParameters"]["nb_epoch"] = nb_epoch |> UInt
    end
    if cutoff_radius != 0.0f0
        conf["TrainingParameters"]["cutoff_radius"] = cutoff_radius
    end
    if name != 0
        conf["TrainingParameters"]["name"] = name
    end
    if learning_rate != 0.
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
    train(training_parameters, auxiliary_parameters)
end

function evaluate_model(model::StatefulLuxLayer, data,
        training_parameters::TrainingParameters)
    (; value, time) = @timed filter(hausdorff_metric.(
        data, Ref(model), Ref(training_parameters))) do x
        !isinf(x)
    end |> mean
    (; metric = value, time)
end
function extract_model(model_serilized::SerializedModel)::StatefulLuxLayer
    model = model_serilized.model()
    StatefulLuxLayer(model,
        model_serilized.weights,
        Lux.initialstates(MersenneTwister(42), model))
end

function evaluate_model(name::String, data, training_parameters::TrainingParameters)
    model = "$(homedir())/datasets/models/$name" |> deserialize |> extract_model
    evaluate_model(model, data, training_parameters)
end

function evaluate_model(names::AbstractArray{String}, tr::TrainingParameters,
        directories::AuxiliaryParameters)
    (; test_data) = get_dataset(tr, directories)
    test_data = test_data[1:10] .|> TreeTrainingData
    evaluate_model.(names, Ref(test_data), Ref(tr)) |> StructArray
end

function evaluate_model(names::AbstractArray{String})
    conf = TOML.parsefile(params_file)
    evaluate_model(names, read_from_TOML(TrainingParameters, conf),
        read_from_TOML(AuxiliaryParameters, conf))
end
@main
