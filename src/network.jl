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
using MLUtils
using Logging
using StaticArrays
using RegionTrees


function load_data(T::Type{<:Number}, name::String)
    TrainingData{T}(extract_balls(T, read("$name.pdb", PDB)), load("$name.off"))
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
