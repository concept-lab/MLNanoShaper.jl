module MLNanoShaper
export DeepSet,model,train, TrainingData,ModelInput,generate_data,ModelInput,load_data

using StructArrays
using GLMakie
using GeometryBasics
using BioStructures
using Reexport

# include("Import.jl")
# @reexport using .Import

include("network.jl")
# include("generate_data.jl")
end
