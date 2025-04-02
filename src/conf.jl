"""
	AuxiliaryParameters

The variables that do not influence the outome of the training run.
This include the nb_epoch.
"""
struct AuxiliaryParameters
    data_dir::String
    log_dir::String
    model_dir::String
    nb_epoch::UInt
    save_periode::Int
    on_gpu::Bool
end
unpact_dict(T::Type, x::AbstractDict{Symbol}) = T(getindex.(Ref(x), fieldnames(T))...)
"""
	TrainingParameters

The training parameters used in the model training.
Default values are in the param file.
The training is deterministric. Theses values are hased to determine a training run
"""
struct TrainingParameters
    name::String
    scale::Float32
    cutoff_radius::Float32
    train_test_split::Float64
    model::Partial
    data_ids::Vector{Int}
    ref_distance::Float32
    loss::LossType
    learning_rate::Float64
end

function generate_training_name(x::TrainingParameters, epoch::Integer)
    "$(x.model().name)_$(x.name)_$(Dates.format(now(),"yyyy-mm-dd"))_epoch_$(epoch)_$(hash(x))"
end

function generate_training_name(x::TrainingParameters)
    "$(x.model().name)_$(x.name)_$(Dates.format(now(),"yyyy-mm-dd"))_$(hash(x))"
end

read_from_TOML(T::Type) = read_from_TOML(T, TOML.parsefile(params_file))

function read_from_TOML(::Type{TrainingParameters}, conf::AbstractDict)
    conf = conf["TrainingParameters"]
    conf = Dict(Symbol.(keys(conf)) .=> values(conf))
    conf[:model_kargs] = Dict(Symbol.(keys(conf[:model_kargs])) .=>
        values(conf[:model_kargs]))
    conf[:model] = Partial(getproperty(MLNanoShaperRunner, Symbol(conf[:model]));
        cutoff_radius = Float32(conf[:cutoff_radius]), conf[:model_kargs]...)
    conf[:loss] = get_loss_type(conf[:loss] |> Symbol)
    unpact_dict(TrainingParameters, conf)
end

function read_from_TOML(::Type{AuxiliaryParameters}, conf::AbstractDict)
    conf = conf["AuxiliaryParameters"]
    conf = Dict(Symbol.(keys(conf)) .=> values(conf))
    unpact_dict(AuxiliaryParameters, conf)
end

const params_file = "$( dirname(dirname(@__FILE__)))/param/param.toml"
