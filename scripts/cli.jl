using ArgParse
using MLNanoShaper: params_file, _train, read_from_TOML, TrainingParameters,
                    AuxiliaryParameters
using Logging
using TerminalLoggers: TerminalLogger
using TOML

function parse_cli_args()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--nb_epoch"
        help = "the number of epochs to compute"
        arg_type = Int
        default = 0
        "--batch_size"
        help=" the size of the batch, must be configured in function of VRAM size"
        default = 0
        arg_type=Int
        "--model", "-m"
        help = "the model name"
        arg_type = String
        default = ""
        "--van_der_waals_channel"
        help = "whether to use van der Waals channel"
        action = :store_true
        "--smoothing"
        help = "whether to enforce smoothing"
        action = :store_true
        "--nb_data_points"
        help = "the number of proteins in the dataset to use"
        arg_type = Int
        default = 0
        "--name", "-n"
        help = "name of the training run"
        arg_type = String
        default = ""
        "--cutoff_radius", "-c"
        help = "the cutoff_radius used in training"
        arg_type = Float32
        default = 0.0f0
        "--ref_distance"
        help = "the reference distance (in A) used to rescale distance to surface in loss"
        arg_type = Float32
        default = 0.0f0
        "--loss"
        help = "the loss function"
        arg_type = String
        default = "categorical"
        "--learning_rate", "-l"
        help = "the learning rate used by the model in training"
        arg_type = Float64
        default = 1e-5
        "--on_gpu", "-g"
        help = "should we do the training on the gpu"
        action = :store_true
    end

    parsed_args = parse_args(s)
    return Dict{Symbol, Any}(Symbol(k) => v for (k, v) in parsed_args)
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
function _main(; nb_epoch::Int = 0,
        batch_size::Int=0,
        model::String = "",
        van_der_waals_channel::Bool = true,
        smoothing::Bool = true,
        nb_data_points::Int = 0,
        name::String = "",
        cutoff_radius::Float32 = 0.0f0,
        ref_distance::Float32 = 00.0f0,
        loss::String = "categorical",
        learning_rate::Float64 = 1e-5,
        on_gpu::Bool = false)
    global_logger(TerminalLogger())
    conf = TOML.parsefile(params_file)
    if nb_epoch > 0
        conf["AuxiliaryParameters"]["nb_epoch"] = nb_epoch |> UInt
    end
    if batch_size > 0
        conf["TrainingParameters"]["batch_size"] = batch_size
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
    conf["AuxiliaryParameters"]["on_gpu"] = on_gpu
    conf["TrainingParameters"]["model_kargs"] = Dict(
        :van_der_waals_channel => van_der_waals_channel, :smoothing => smoothing,:on_gpu => on_gpu)
    if nb_data_points > 0
        conf["TrainingParameters"]["data_ids"] = conf["TrainingParameters"]["data_ids"][begin:(begin + nb_data_points)]
    end

    training_parameters = read_from_TOML(TrainingParameters, conf)
    auxiliary_parameters = read_from_TOML(AuxiliaryParameters, conf)
    _train(training_parameters, auxiliary_parameters)
end

cli_args = parse_cli_args()
# Call main function with parsed arguments
_main(; cli_args...)

