using Lux
using ConcreteStructs
using GeometryBasics

@concrete
struct DeepSet <: Lux.AbstractExplicitContainerLayer{(:prepross, :postpros)}
    prepross::Any
end

function (f::DeepSet)(set::AbstractSet, ps, st)
    sum(set) do x
        f.prepross(x, ps, st)
    end
end

struct TrainingData
    x::Set{Sphere{Float64}}
    y::Mesh
end

distance2(x::Point3, y::Point3) = sum((x .- y) .^ 2)
distance2(x::Point3, y::Mesh) =
    minimum(y) do y
        distance2(x, y)
    end

function loss(model,ps,st,(x,y))
	d_pred,st = Lux.apply(model,x,ps,st)
	d_pred - distance2(x,y),st, (;)
end

function preprossesing(xyzr::AbstractSet{Sphere{<:Real}}) end
function encoding(pos::Point3, x::Sphere) end
function select_radius(r2::Number, set::AbstractSet{Tuple{Point3{<:Real}, Sphere{<:Real}}})
    filter(set) do pos, sphere
        distance2(pos, sphere.center) <= r2
    end
end

model = Chain(preprossesing,
    Base.Fix1(select_radius, 1.5),
    DeepSet(Chain(encoding, Dense(50 => 30, relu), Dense(30 => 1))))
