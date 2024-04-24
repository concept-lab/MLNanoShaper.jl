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
using MLUtils
using Logging
using StaticArrays
using RegionTrees
using Optimisers
using Statistics
using TensorBoardLogger
using Serialization

"""
	train((train_data,test_data),training_states; nb_epoch)
train the model on the data with nb_epoch
"""
function train((train_data,
            test_data)::Tuple{MLUtils.AbstractDataContainer, MLUtils.AbstractDataContainer},
        training_states::Lux.Experimental.TrainState; nb_epoch, save_periode, params...)
    # serialize("$(homedir())/$(conf["paths"]["model_dir"])/model_0", training_states)
    for epoch in 1:nb_epoch
        (; training_states, losses) = train(train_data, training_states; params...)
        @info "loss" losses
        @info "model" training_states
        # results = test(data, training_states)
        # @info "accuracy" results
        # if epoch % save_periode == 0
        #     serialize("$(homedir())/$(conf["paths"]["model_dir"])/model_$epoch",
        #         training_states)
        # end
    end
end
function train(data,
        training_states::Lux.Experimental.TrainState; params...)
    losses_v = Float32[]
    for d in data
        (; training_states, losses) = train(d, training_states; params...)
        append!(losses_v, losses)
    end
    (; training_states, losses = mean(losses_v))
end
function train((; atoms, skin)::TrainingData{Float32},
        training_states::Lux.Experimental.TrainState; scale::Float32,
        r::Float32)
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
    losses_v = Float32[]

    for point in points
        @info "training" point
        atoms_neighboord = select_radius(r, point, atoms)
        if length(atoms_neighboord) == 0
            continue
        end
        grads, losses, _, training_states = Lux.Experimental.compute_gradients(AutoZygote(),
            loss,
            (; point, atoms = atoms_neighboord, skin),
            training_states)
        push!(losses_v, losses)
        training_states = Lux.Experimental.apply_gradients(training_states, grads)
    end
    (; training_states, losses = losses_v)
end

function test(test_data, training_states) end

"""
	load_data(T, name::String)

Load a `TrainingData{T}` from current directory.
You should have a pdb and an off file with name `name` in current directory.
"""
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
function train()
    data = splitobs(mapobs(shuffle(MersenneTwister(42),
            conf["protein"]["list"])) do name
            load_data(Float32, "$datadir/$name")
        end; at = 0.8)
    with_logger(TBLogger("$(homedir())/$(conf["paths"]["model_dir"])")) do
        train(data,
            Lux.Experimental.TrainState(MersenneTwister(42), model,
                Adam(0.01));
            nb_epoch = 1,
            save_periode = 1,
            r = 1.5f0,
            scale = 1.0f0)
    end
end
