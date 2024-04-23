using Lux
using LinearAlgebra: NumberArray
using ConcreteStructs
using TOML
using GeometryBasics
using ADTypes
using Random
using LinearAlgebra
using FileIO
using Zygote
using Adapt
using MLUtils
using Logging
using SimpleChains: static

@concrete struct DeepSet <: Lux.AbstractExplicitContainerLayer{(:prepross,)}
    prepross
    init
end

function (f::DeepSet)(set::AbstractSet{T}, ps, st) where {T}
    sum(set; init = f.init) do arg
        first(f.prepross(arg, ps, st))
    end, st
end

struct TrainingData{T <: Number}
    atoms::Set{Sphere{T}}
    skin::GeometryBasics.Mesh
end

function load_data(T::Type{<:Number}, name::String)
    TrainingData{T}(extract_balls(T, read("$name.pdb", PDB)), load("$name.off"))
end

struct ModelInput{T <: Number}
    point::Point3{T}
    atoms::Set{Sphere{T}}
end

struct PreprocessData{T <: Number}
    dot::T
    r_1::T
    r_2::T
    d_1::T
    d_2::T
end

function cut(cut_radius::Number, r::Number)
    if r >= cut_radius
        0
    else
        (1 + cos(π * r / cut_radius)) / 2
    end
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
    encoded = 2 .* ((1 .+ (x.dot .- dotsₛ)) ./ 2) .^ ζ .*
              exp.(-η .* ((x.d_1 + x.d_2) / 2 .- Dₛ) .^ 2) .*
              cut(l.cut_distance, x.d_1) .*
              cut(l.cut_distance, x.d_2)
    x = vcat(vec(encoded), [(x.r_1 + x.r_2) / 2, abs(x.r_1 - x.r_2)])
    x, st
end

distance2(x::Point3{T}, y::Point3{T}) where {T <: Number} = sum((x .- y) .^ 2)
distance2(x::Point3{Float32}, y::GeometryBasics.Mesh) =
    minimum(coordinates(y)) do y
        distance2(x, y)
    end

function loss(model, ps, st, (; point, atoms, skin))
    d_pred, st = Lux.apply(model, ModelInput(point, atoms), ps, st)
    only(d_pred) - distance2(point, skin), st, (;)
end

function train((; atoms, skin)::TrainingData{Float32},
        training_states::Lux.Experimental.TrainState)::Lux.Experimental.TrainState
    scale = 0.5f0
    r2 = 1.5f0^2

    min_coordinate::Vector{Float32} = mapreduce(collect,
        (x, y) -> min.(x, y),
        coordinates(skin))
    max_coordinate::Vector{Float32} = mapreduce(collect,
        (x, y) -> max.(x, y),
        coordinates(skin))
    points::Vector{Point3{Float32}} = filter(Iterators.product(range.(min_coordinate,
        max_coordinate,
        ; step = scale)...) .|> Point3) do point
        distance2(point, skin) < r2
    end

    for point in points
        @info "training" point
        grads, _, _, training_states = Lux.Experimental.compute_gradients(AutoZygote(),
            loss,
            (; point, atoms, skin),
            training_states)
        training_states = Lux.Experimental.apply_gradients(training_states, grads)
    end
    training_states
end

function train(data::MLUtils.AbstractDataContainer,
        training_states::Lux.Experimental.TrainState)
    for d in data
        training_states = train(d, training_states)
    end
    training_states
end

function preprocessing((; point, atoms)::ModelInput)
    map(Iterators.product(atoms, atoms)) do (atom1, atom2)::Tuple{Sphere, Sphere}
        d_1 = sqrt(distance2(point, atom1.center))
        d_2 = sqrt(distance2(point, atom2.center))
        dot = (atom1.center - point) ⋅ (atom2.center - point) / (d_1 * d_2 + 1.0f-8)
        PreprocessData(dot, atom1.r, atom2.r, d_1, d_2)
    end |> Set
end
function select_radius(r2::Number, (; point, atoms)::ModelInput)
    ModelInput(point, filter(atoms) do sphere
        distance2(point, sphere.center) <= r2
    end)
end
a = 5
b = 5
adaptator = ToSimpleChainsAdaptor((static(a * b + 2),))
chain = Chain(Dense(a * b + 2 => 10, relu),
    Dense(10 => 10, relu), Dense(10 => 1))
model = Lux.Chain(Base.Fix1(select_radius, 1.5f0), preprocessing,
    DeepSet(Chain(Encoding(a, b, 1.5f0), adapt(adaptator, chain)),
        zeros32(1)))

data = mapobs(shuffle(MersenneTwister(42),
    conf["protein"]["list"])) do name
    load_data(Float32, "$datadir/$name")
end
