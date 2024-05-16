using TOML
using Lux
using MLNanoShaperRunner

"""
	Auxiliary_parameters

The variables that do not influence the outome of the training run.
This include the nb_epoch.
"""
struct Auxiliary_parameters
    data_dir::String
    log_dir::String
    model_dir::String
    nb_epoch::UInt
    save_periode::Int
end
unpact_dict(T::Type, x::AbstractDict{Symbol}) = T(getindex.(Ref(x), fieldnames(T))...)

"""
	Training_parameters

The training parameters used in the model training.
Default values are in the param file.
The training is deterministric. Theses values are hased to determine a training run
"""
struct Training_parameters
    name::String
    scale::Float32
    cutoff_radius::Float32
    train_test_split::Float64
    model::Lux.AbstractExplicitLayer
    data_ids::Vector{Int}
end

function generate_training_name(x::Training_parameters, epoch::Integer)
	"$(x.model.name)_$(x.name)_epoch_$(epoch)_$(hash(x))"
end

function generate_training_name(x::Training_parameters)
	"$(x.model.name)_$(x.name)_$(hash(x))"
end

read_from_TOML(T::Type) = read_from_TOML(T, TOML.parsefile(params_file))

function read_from_TOML(::Type{Training_parameters}, conf::AbstractDict)
    conf = conf["Training_parameters"]
    conf = Dict(Symbol.(keys(conf)) .=> values(conf))

	conf[:model] = getproperty(MLNanoShaperRunner, Symbol(conf[:model]))(;cutoff_radius=conf[:cutoff_radius])
    unpact_dict(Training_parameters, conf)
end

function read_from_TOML(::Type{Auxiliary_parameters}, conf::AbstractDict)
    conf = conf["Auxiliary_parameters"]
    conf = Dict(Symbol.(keys(conf)) .=> values(conf))
    unpact_dict(Auxiliary_parameters, conf)
end

const params_file = "$( dirname(dirname(@__FILE__)))/param/param.toml"
