module MLNanoShaper
using Configurations
using TerminalLoggers: TerminalLogger
using TerminalLoggers
using Logging: global_logger
using Logging
using LoggingExtras: shouldlog
using LoggingExtras
using Logging
using ProgressLogging
using TensorBoardLogger

using Zygote
using ADTypes
using Lux
using LuxCUDA
using MLUtils
using Optimisers
using ChainRulesCore

using Statistics
using Random
using StaticArrays

using StaticArrays: reorder
using LinearAlgebra: NumberArray
using StructArrays
using ConcreteStructs
using Static
using GeometryBasics
using Meshing
using NearestNeighbors
using LinearAlgebra: NumberArray
using LinearAlgebra
using Folds

using Dates
using TOML
import BioStructures: PDBFormat, downloadpdb
using FileIO
using Serialization
using Reexport

using LoopVectorization
using Octavian
using PrecompileTools
using MLNanoShaperRunner
using MLNanoShaperRunner: Partial
@reexport using MLNanoShaperRunner.Import

export train

include("optimiser.jl")
include("loss.jl")
include("conf.jl")
include("logging.jl")
include("preprocessing.jl")
include("training.jl")
include("generate_data.jl")
# @setup_workload begin
#     inputs = (Batch([Point3f(0,0,0)]),RegularGrid([Sphere(Point3f(0,0,0),1f0),Sphere(Point3f(0,0,0),1f0)] |> StructVector,3f0))
#     model = MLNanoShaperRunner.tiny_soft_max_angular_dense()
#     preprocessing = MLNanoShaperRunner.get_preprocessing(model)
#     _model = MLNanoShaperRunner.drop_preprocessing(model)
#     ps,st = Lux.setup(MersenneTwister(42),model) |> cu
#     @compile_workload begin
#         preprocessed_data,_ = preprocessing(inputs,ps,st)
#         jacobian(ps) do ps
#             _model(preprocessed_data,ps,st) |> first 
#         end
#     end
# end

end
