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
    batch_size::Int
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
function Base.hash(x::TrainingParameters, h::UInt)
    h = hash(x.name, h)
    h = hash(x.scale, h)
    h = hash(x.cutoff_radius, h)
    h = hash(x.train_test_split, h)
    h = hash(x.model, h)
    h = hash(x.data_ids, h)
    h = hash(x.ref_distance, h)
    h = hash(x.loss, h)
    h = hash(x.learning_rate, h)
    return h
end

function generate_training_name(x::TrainingParameters, epoch::Integer)
    "$(x.model().name)_$(x.name)_$(epoch)_$(hash(x))"
end

function generate_training_name(x::TrainingParameters)
    # @info hash(x.model()) hash(x.loss) hash(x.data_ids) hash(x.name)
    "$(x.model().name)_$(x.name)_$(hash(x))"
end
function find_latest_epoch_file(directory::String, training_parameters::TrainingParameters)
    training_name_prefix = "$(training_parameters.model().name)_$(training_parameters.name)"
    hash_value = hash(training_parameters)
    files = readdir(directory)
    max_epoch = -1
    latest_file = nothing

    for file in files
        # Split into name and extension

        # The pattern to match the name part
        pattern = Regex("^$(training_name_prefix)_(\\d+)_$(hash_value)\$")
        m = match(pattern, file)
        if m !== nothing
            epoch = parse(Int, m.captures[1])
            if epoch > max_epoch
                max_epoch = epoch
                latest_file = file
            end
        end
    end

    return latest_file,max_epoch
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
