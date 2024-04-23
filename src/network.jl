using Lux
using Base: AbstractArrayOrBroadcasted
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
using StaticArrays
using RegionTrees


function load_data(T::Type{<:Number}, name::String)
    TrainingData{T}(extract_balls(T, read("$name.pdb", PDB)), load("$name.off"))
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

struct DensityRefinery <: AbstractRefinery
    nb_points::Int
    pos::Function
end

function RegionTrees.refine_data(r::DensityRefinery, cell, indices)
    boundary = child_boundary(cell, indices)
    filter(cell.data) do point
        boundary.origin <= r.pos(point) <= boundary.origin + boundary.widths
    end
end
function RegionTrees.needs_refinement(refinery::DensityRefinery, cell)
    length(cell.data) > refinery.nb_points
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
	encoded = 2 .* (max.(0,(1 .+ (x.dot .- dotsₛ))) ./ 2) .^ ζ .*
              exp.(-η .* ((x.d_1 + x.d_2) / 2 .- Dₛ) .^ 2) .*
              cut(l.cut_distance, x.d_1) .*
              cut(l.cut_distance, x.d_2)
    x = vcat(vec(encoded), [(x.r_1 + x.r_2) / 2, abs(x.r_1 - x.r_2)])
    x, st
end

distance2(x::Point3{T}, y::Point3{T}) where {T <: Number} = sum((x .- y) .^ 2)
distance2(x::Point3{Float32}, y::GeometryBasics.Mesh) = distance2(x, coordinates(y))

function distance2(x::Point3{T},
        y::Cell{<:AbstractVector{Point3{T}}, 3, T, <:Any}) where {T}
    distance2(x, findleaf(y, x).data)
end

function distance2(x::Point3{T}, y::AbstractArray{Point3{T}}) where {T}
    minimum(y) do y
        distance2(x, y)
    end
end

function loss(model, ps, st, (; point, atoms, skin))
    d_pred, st = Lux.apply(model, ModelInput(point, atoms), ps, st)
    only(d_pred) - distance2(point, skin), st, (;)
end
function box_coordinate(f, collection)
    mapreduce(collect, (x, y) -> f.(x, y), collection)
end
function train((; atoms, skin)::TrainingData{Float32},
        training_states::Lux.Experimental.TrainState)::Lux.Experimental.TrainState
    scale = 0.5f0
    r = 1.5f0
    @info "start training"

    min_coordinate = box_coordinate(min, coordinates(skin)) |> SVector{3}
    max_coordinate = box_coordinate(max, coordinates(skin)) |> SVector{3}
    skin = Cell(min_coordinate, max_coordinate, coordinates(skin))
    adaptivesampling!(skin, DensityRefinery(10000, identity))

    atoms = Cell(min_coordinate, max_coordinate, collect(atoms))
    adaptivesampling!(atoms, DensityRefinery(100, sph -> sph.center))

    points::Vector{Point3{Float32}} = filter(Iterators.product(range.(min_coordinate,
        max_coordinate,
        ; step = scale)...) .|> Point3) do point
        distance2(point, skin) < r^2
    end

    for point in points
		@info "training" point
        atoms_neighboord = select_radius(r, point, atoms)
        if length(atoms_neighboord) == 0
            continue
        end
        grads, _, _, training_states = Lux.Experimental.compute_gradients(AutoZygote(),
            loss,
            (; point, atoms = atoms_neighboord, skin),
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
    end
end

function filter_cells(test::Function, cell::Cell)
    res = Cell[]
    queue = [cell]
    while !isempty(queue)
        current = pop!(queue)
        if !test(current)
            continue
        end
        push!(res, current)
        if !isleaf(current)
            append!(queue, children(current))
        end
    end
    res
end

function select_radius(cut_radius::T,
        point::Point3{T},
        atoms::Cell{<:AbstractVector{Sphere{T}}, 3}) where {T}

    atoms = filter_cells(atoms) do node::Cell
        center = node.boundary.origin + node.boundary.widths / 2
        center = Point3(center...)
        widths = node.boundary.widths
        distance2(point, center) <= (cut_radius + sum(widths) / 2)^2
    end

    atoms = mapreduce(vcat, atoms) do node::Cell
        if isleaf(node)
            filter(node.data) do (; center)::Sphere
                distance2(point, center) <= (2 * cut_radius)^2
            end
        else
            Sphere{T}[]
        end
    end
    atoms
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
