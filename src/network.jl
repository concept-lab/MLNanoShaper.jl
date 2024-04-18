using Lux
using ConcreteStructs
using GeometryBasics
using ADTypes
using Random
using LinearAlgebra

@concrete struct DeepSet <: Lux.AbstractExplicitContainerLayer{(:prepross,)}
    prepross
end

function (f::DeepSet)(set::AbstractSet, ps, st)
    sum(set) do x
        f.prepross(x, ps, st)
    end
end

struct TrainingData
    atoms::Set{Sphere{Float64}}
    skin::GeometryBasics.Mesh
end

struct Input
    point::Point3
    atoms::Set{Sphere{Float64}}
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
@concrete struct Encoding <: Lux.AbstractExplicitLayer
    n_dotₛ::Int
    n_Dₛ::Int
    cut_distance
end

function Lux.initialparameters(::AbstractRNG, l::Encoding)
    (dotsₛ = reshape(collect(range(0, 1; length = l.n_dotₛ)), 1, :),
        Dₛ = reshape(collect(range(0, l.cut_distance; length = l.n_Dₛ)), :, 1),
		η =  reshape([1 / l.n_Dₛ],1,1),
		ζ =  reshape([1 / l.n_dotₛ],1,1))
end
Lux.initialstates(::AbstractRNG, l::Encoding) = (;)

function (l::Encoding{T})(x::PreprocessData{T}, ps, st) where {T}
    encoded = 2 .* ((1 .+ (x.dot .- ps.dotsₛ)) ./ 2) .^ ps.ζ .*
              exp.(-ps.η .* ((x.d_1 + x.d_2) / 2 .- ps.Dₛ) .^ 2) .*
              cut(l.cut_distance, x.d_1) .*
              cut(l.cut_distance, x.d_2)
    x = vcat(vec(encoded), [(x.r_1 + x.r_2) / 2, abs(x.r_1 - x.r_2)])
	x,st
end

distance2(x::Point3, y::Point3) = sum((x .- y) .^ 2)
distance2(x::Point3, y::GeometryBasics.Mesh) =
    minimum(y) do y
        distance2(x, y)
    end

function loss(model, ps, st, (pos, atoms, skin))
    d_pred, st = Lux.apply(model, Input(pos, atoms), ps, st)
    d_pred - distance2(pos, skin), st, (;)
end

function train(data::TrainingData,
        training_states::Lux.Experimental.TrainState)
    scale = 0.1
    r2 = 1.5^2
    points = filter(Point3.(Iterators.product(0:scale:10,
        0:scale:10,
        0:scale:10))) do point
        distance2(point, data.skin) < r2
    end

    for point in points
        grads, _, _, training_states = Lux.Experimental.compute_gradients(AutoZygote(),
            loss,
            (point, data.atoms, data.skin),
            training_states)
        training_states = Lux.Experimental.apply_gradients(training_states, grads)
    end
end
function preprossesing(data::Input)
    (; point, atoms) = data
    map(Iterators.product(atoms, atoms)) do (atom1, atom2)::Tuple{Sphere, Sphere}
        d_1 = sqrt(distance2(point, atom1.center))
        d_2 = sqrt(distance2(point, atom2.center))
        dot = (atom1.center - point) ⋅ (atom2.center - point) / (d_1 * d_2)
        PreprocessData(dot, atom1.r, atom2.r, d_1, d_2)
    end |> Set
end
function select_radius(r2::Number, data::Input)
    (; point, atoms) = data
    Input(point, filter(atoms) do sphere
        distance2(point, sphere.center) <= r2
    end)
end
a = 5
b= 5
model = Lux.Chain(Base.Fix1(select_radius, 1.5), preprossesing,
    DeepSet(Lux.Chain(Encoding(a, b, 1.5),
        Dense(a*b +2=> 30, relu),
        Dense(30 => 10, relu))), Dense(10 => 1))
