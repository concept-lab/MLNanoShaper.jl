using Lux
using ConcreteStructs
using GeometryBasics
using Random
using SimpleChains: static
using Adapt

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
struct ModelInput{T <: Number}
    point::Point3{T}
    atoms::Vector{Sphere{T}} #Set
end

struct PreprocessData{T <: Number}
    dot::T
    r_1::T
    r_2::T
    d_1::T
    d_2::T
end
struct Encoding{T <: Number} <: Lux.AbstractExplicitLayer
    n_dotₛ::Int
    n_Dₛ::Int
    cut_distance::T
end

function Lux.initialparameters(::AbstractRNG, l::Encoding{T}) where {T}
    (dotsₛ = reshape(collect(range(T(0), T(1); length = l.n_dotₛ)), 1, :),
        Dₛ = reshape(collect(range(T(0), l.cut_distance; length = l.n_Dₛ)), :, 1),
        η = reshape([T(1) / l.n_Dₛ], 1, 1),
        ζ = reshape([T(1) / l.n_dotₛ], 1, 1))
end
Lux.initialstates(::AbstractRNG, l::Encoding) = (;)

function (l::Encoding{T})(x::PreprocessData{T}, (; dotsₛ, η, ζ, Dₛ), st) where {T}
    encoded = 2 .* (max.(0, (1 .+ (x.dot .- dotsₛ))) ./ 2) .^ ζ .*
              exp.(-η .* ((x.d_1 + x.d_2) / 2 .- Dₛ) .^ 2) .*
              cut(l.cut_distance, x.d_1) .*
              cut(l.cut_distance, x.d_2)
    x = vcat(vec(encoded), [(x.r_1 + x.r_2) / 2, abs(x.r_1 - x.r_2)])
    x, st
end

function cut(cut_radius::Number, r::Number)
    if r >= cut_radius
        0
    else
        (1 + cos(π * r / cut_radius)) / 2
    end
end

function preprocessing((; point, atoms)::ModelInput)
    map(Iterators.product(atoms, atoms)) do (atom1, atom2)::Tuple{Sphere, Sphere}
        d_1 = sqrt(distance2(point, atom1.center))
        d_2 = sqrt(distance2(point, atom2.center))
        dot = (atom1.center - point) ⋅ (atom2.center - point) / (d_1 * d_2 + 1.0f-8)
        PreprocessData(dot, atom1.r, atom2.r, d_1, d_2)
    end
end

a = 5
b = 5
adaptator = ToSimpleChainsAdaptor((static(a * b + 2),))
chain = Chain(Dense(a * b + 2 => 10, relu),
    Dense(10 => 1))
model = Lux.Chain(preprocessing,
    DeepSet(Chain(Encoding(a, b, 1.5f0), adapt(adaptator, chain))))

# data = mapobs(shuffle(MersenneTwister(42),
#     conf["protein"]["list"])) do name
#     load_data(Float32, "$datadir/$name")
# end
