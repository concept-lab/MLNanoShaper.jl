using Comonicon
using Serialization
using Random
using Configurations
using Logging: global_logger
using TerminalLoggers: TerminalLogger
"""
    train

Train a model.
Parameters are specified in the `param/param.toml` file.
The folowing parameters can be overided.

# Options
- `-e, --nb-epoch <Int>`: the number of epoch to compute.
- `-m, --model <String>`: the model name. Can be anakin.
- `-n, --name <String>`: name of the training run
- `-c, --cutoff-radius <Float32>`: the cutoff_radius used in training

"""
@option struct ModelArgs
    van_der_wal_channel::Bool = false
end
@cast function train(; nb_epoch::Int = 0,
        model::String = "",
        model_kargs::ModelArgs,
        nb_data_points::Int = 0,
        name::String = "",
        cutoff_radius::Float32 = 0.0f0)
    global_logger(TerminalLogger())
    conf = TOML.parsefile(params_file)
    if nb_epoch > 0
        conf["Auxiliary_parameters"]["nb_epoch"] = nb_epoch |> UInt
    end
    if cutoff_radius != 0.0f0
        conf["Training_parameters"]["cutoff_radius"] = cutoff_radius
    end
    if name != 0
        conf["Training_parameters"]["name"] = name
    end
    if model != ""
        conf["Training_parameters"]["model"] = model
    end
    if model_kargs != ModelArgs()
        conf["Training_parameters"]["model_kargs"] = Configurations.to_dict(model_kargs)
    end
    if nb_data_points > 0
        conf["Training_parameters"]["data_ids"] = conf["Training_parameters"]["data_ids"][begin:(begin + nb_data_points)]
    end

    training_parameters = read_from_TOML(Training_parameters, conf)
    auxiliary_parameters = read_from_TOML(Auxiliary_parameters, conf)
    train(training_parameters, auxiliary_parameters)
end

function evaluate_model(model::StatefulLuxLayer, data,
        training_parameters::Training_parameters)
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

function evaluate_model(name::String, data, training_parameters::Training_parameters)
    model = "$(homedir())/datasets/models/$name" |> deserialize |> extract_model
    evaluate_model(model, data, training_parameters)
end

function evaluate_model(names::AbstractArray{String}, tr::Training_parameters,
        directories::Auxiliary_parameters)
    (; test_data) = get_dataset(tr, directories)
    test_data = test_data[1:10] .|> TreeTrainingData
    evaluate_model.(names, Ref(test_data), Ref(tr)) |> StructArray
end

function evaluate_model(names::AbstractArray{String})
    conf = TOML.parsefile(params_file)
    evaluate_model(names, read_from_TOML(Training_parameters, conf),
        read_from_TOML(Auxiliary_parameters, conf))
end
@main
