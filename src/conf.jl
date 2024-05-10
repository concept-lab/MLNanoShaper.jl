using TOML
using Lux

"""
	Auxiliary_parameters

The variables that do not influence the outome of the training run.
This include the nb_epoch.
"""
struct Auxiliary_parameters
	data_dir::String
	log_dir::String
	models_dirs::String
	nb_epoch::UInt
	save_periode::Int
end

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
	data_ids::Vector{String}
end

function generate_training_name(x::Training_parameters,epoch::Integer)
	"$(x.name)_epoch_$(epoch)_$(hash(x))"
end


read_from_TOML(T::Type) = read_from_TOML(T,TOML.parsefile(params_file))

function read_from_TOML(::Type{Training_parameters},conf::AbstractDict)
	conf = conf["Training_parameters"]
	conf = Dict( Symbol.(keys(conf)) .=> values(conf)) 
	conf[:model] = getfield(Main,Symbol(conf[:model]))()
	Training_parameters(;conf...)
end

function read_from_TOML(::Type{Auxiliary_parameters},conf::AbstractDict)
	conf = Dict( Symbol.(keys(conf)) .=> values(conf)) 
	Auxiliary_parameters(;conf["Auxiliary_parameters"]...)
end

const params_file = "$( dirname(dirname(@__FILE__)))/param/param.toml"
