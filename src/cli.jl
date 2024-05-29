using Comonicon
using Serialization
using Random
Option{T} = Union{T, Nothing}
"""
    train

Train a model.
Parameters are specified in the `param/param.toml` file.
The folowing parameters can be overided.

# Options
- `-e, --nb-epoch=Uint`; the number of epoch to compute.
- `-m, --model=String`; the model name. Can be anakin.

"""
@cast function train(; nb_epoch::Option{UInt} = nothing, model::Option{String} = nothing,
        nb_data_points::Option{UInt} = nothing, name::Option{String} = nothing, cutoff_radius::Option{Float32} = nothing)
    conf = TOML.parsefile(params_file)
    if !isnothing(nb_epoch)
        conf["Training_parameters"]["nb_epoch"] = nb_epoch
    end
    if !isnothing(cutoff_radius)
        conf["Training_parameters"]["cutoff_radius"] = cutoff_radius
    end
    if !isnothing(name)
        conf["Training_parameters"]["name"] = name
    end
    if !isnothing(model)
        conf["Training_parameters"]["model"] = model
    end
    if !isnothing(nb_data_points)
        conf["Training_parameters"]["data_ids"] = conf["Training_parameters"]["data_ids"][begin:(begin + nb_data_points)]
    end

    training_parameters = read_from_TOML(Training_parameters, conf)
    auxiliary_parameters = read_from_TOML(Auxiliary_parameters, conf)
    @info "Starting training"
    train(training_parameters, auxiliary_parameters)
    @info "Stop training"
end

function evaluate_model(model::StatefulLuxLayer, data,
        training_parameters::Training_parameters)
    (; value, time) = @timed filter(hausdorff_metric.(
        data, Ref(model), Ref(training_parameters))) do x
        !isinf(x)
    end |> mean
    (; metric = value, time)
end

function evaluate_model(name::String, data, training_parameters::Training_parameters)
    model_serilized::SerializedModel = deserialize("$(homedir())/datasets/models/$name")
    model = model_serilized.model()
    model = StatefulLuxLayer(model,
        model_serilized.weights,
        Lux.initialstates(MersenneTwister(42), model))
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
