using Lux
using ConcreteStructs
using GeometryBasics

@concrete struct DeepSet <: Lux.AbstractExplicitContainerLayer{(:prepross,)}
    prepross
end

function (f::DeepSet)(set::AbstractArray{T}, ps, st) where {T}
    sum(set) do arg
        first(f.prepross(arg, ps, st))
    end, st
end

struct TrainingData{T <: Number}
    atoms::Set{Sphere{T}}
    skin::GeometryBasics.Mesh
end
