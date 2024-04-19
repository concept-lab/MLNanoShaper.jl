module MLNanoShaper
export DeepSet,model,train, TrainingData,ModelInput,generate_data

using StructArrays
using GLMakie
using GeometryBasics
using BioStructures
using Reexport

include("Inport.jl")
@reexport using .Inport

include("network.jl")
include("generate_data.jl")
end
