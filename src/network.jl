using Lux
using ConcreteStructs
using GeometryBasics
using ADTypes

@concrete
struct DeepSet <: Lux.AbstractExplicitContainerLayer{(:prepross,)}
    prepross
end

function (f::DeepSet)(set::AbstractSet, ps, st)
    sum(set) do x
        f.prepross(x, ps, st)
    end
end

struct TrainingData
    atoms::Set{Sphere{Float64}}
    skin::Mesh
end

struct Input
    pos::Point3
    atoms::Set{Sphere{Float64}}
end

struct PreprocessData{T <: Number}
    dot::T
    r_1::T
    r_2::T
    d_1::T
    d_2::T
end

struct Encoding{T <: Float64} <: Lux.AbstractExplicitLayer{(:dotₛ, :Rₛ, :ζ, :η)}
    n_dotₛ::T
    n_Rₛ::T
	cut_radius::T
end


cut(cut_radius::Number,r::Number) = if r >= cut_radius 0 else (1 + cos(π*r/cut_radius))/2

function (l::Encoding{T})(x::PreprocessData{T}, ps, _) where {T}
	2 .*((1 .+ (x.dot .-ps.dotₛ))/2).^ps.ζ * exp.( -ps.η*((x.r_1 + x.r_2)/2 .- ps.Rₛ).^2)*cut(l.cut_radius,x.r_1)*cut(l.cut_radius,x.r_2) |> collect
end

distance2(x::Point3, y::Point3) = sum((x .- y) .^ 2)
distance2(x::Point3, y::Mesh) =
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
    points = filter(Point3.(Iterators.product(0:scale:10, 0:scale:10, 0:scale:10))) do point
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
    map(Iterators.product(atoms, atoms)) do (center_1, r_1), (center_2, r_2)
        d_1 = sqrt(distance2(point, center_1))
        d_2 = sqrt(distance2(point, center_2))
        dot = (center_1 - point) ∘ (center_2 - point) / (d_1 * d_2)
        PreprocessData(dot, r_1, r_2, d_1, d_2)
    end
end
function select_radius(r2::Number, set::AbstractSet{Tuple{Point3{<:Real}, Sphere{<:Real}}})
    filter(set) do pos, sphere
        distance2(pos, sphere.center) <= r2
    end
end

model = Chain(preprossesing,
    Base.Fix1(select_radius, 1.5),
	DeepSet(Chain(Encoding(5,10,1.5), Dense(50 => 30, relu), Dense(30 => 10, relu))), Dense(10 => 1))
