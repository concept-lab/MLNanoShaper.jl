using Lux
using Distributed: preduce
using StaticArrays: reorder
using LoggingExtras: shouldlog
using LinearAlgebra: NumberArray
using ConcreteStructs
using TOML
using GeometryBasics
using ADTypes
using Random
using LinearAlgebra
using FileIO
using Zygote
using MLUtils
using Logging
using StaticArrays
using Optimisers
using Statistics
using TensorBoardLogger
using Serialization
using Meshing
using NearestNeighbors
using StructArrays
using MLNanoShaperRunner
using Distributed
using Static

"""
Training information used in model training.
# Fields
- `atoms`: the set of atoms used as model input
- `skin` : the Surface generated by Nanoshaper
"""
struct TrainingData{T <: Number}
    atoms::StructVector{Sphere{T}}
    skin::GeometryBasics.Mesh
end

struct TreeTrainingData{T <: Number}
    atoms::AnnotedKDTree{Sphere{T}, :center, Point3{T}}
    skin::RegionMesh
end
function TreeTrainingData((; atoms, skin)::TrainingData)
    TreeTrainingData(AnnotedKDTree(atoms, static(:center)), RegionMesh(skin))
end

function point_grid(rng::AbstractRNG, atoms_tree::KDTree,
        skin_tree::KDTree{Point3f},
        (; scale,
            cutoff_radius)::Training_parameters)
    (; mins, maxes) = atoms_tree.hyper_rec
    points = first(
        shuffle(
            rng, Iterators.product(range.(mins,
                maxes
                ; step = scale)...) .|> Point3),
        1000)
    Iterators.filter(points) do point
        distance(point, atoms_tree) < cutoff_radius &&
            distance(point, skin_tree) < cutoff_radius
    end
end

function exact_points(
        rng::AbstractRNG, atoms_tree::KDTree, skin_tree::KDTree, (;
            cutoff_radius)::Training_parameters)
    points = first(shuffle(rng, skin_tree.data), 200)
    Iterators.filter(points) do pt
        distance(pt, atoms_tree) < cutoff_radius
    end
end
function generate_data_points(preprocessing::Lux.AbstractExplicitLayer, points,
        (; atoms, skin)::TreeTrainingData{Float32})

    # exact_points_v = exact_points(MersenneTwister(42), atoms.tree, skin, cutoff_radius)
    # points = first(
    # point_grid(MersenneTwister(42), atoms.tree, skin.tree; scale, cutoff_radius), 40)

    mapobs(points) do point::Point3f
        (; point, input = preprocessing((point, atoms)),
            d_real = signed_distance(point, skin))
    end
end
GLobalPreprocessed=@NamedTuple{
                point::Point3f, input::StructArray{PreprocessData{Float32}}, d_real::Float32}
function pre_compute_data_set(f::Function,
        preprocessing,
        data::AbstractVector{<:TreeTrainingData})
    res = Folds.map(data) do d
        points = f(d)
        collect(
            GLobalPreprocessed,
            generate_data_points(preprocessing, points, d))
    end
    reduce(vcat, res)
end
"""
	load_data_pdb(T, name::String)

Load a `TrainingData{T}` from current directory.
You should have a pdb and an off file with name `name` in current directory.
"""
function load_data_pdb(T::Type{<:Number}, name::String)
    TrainingData{T}(extract_balls(T, read("$name.pdb", PDB)), load("$name.off"))
end
"""
	load_data_pqr(T, name::String)

Load a `TrainingData{T}` from current directory.
You should have a pdb and an off file with name `name` in current directory.
"""
function load_data_pqr(T::Type{<:Number}, dir::String)
    TrainingData{T}(getproperty.(read("$dir/structure.pqr", PQR{T}), :pos) |> StructVector,
        load("$dir/triangulatedSurf.off"))
end