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
using LoggingExtras

"""
	train((train_data,test_data),training_states; nb_epoch)
train the model on the data with nb_epoch
"""
function train((train_data,
            test_data)::Tuple{MLUtils.AbstractDataContainer, MLUtils.AbstractDataContainer},
        training_states::Lux.Experimental.TrainState; nb_epoch, save_periode, params...)
    # serialize("$(homedir())/$(conf["paths"]["model_dir"])/model_0", training_states)
    for epoch in 1:nb_epoch
        println("epoch $epoch/$nb_epoch")
        training_states = train(train_data, training_states; params...)
		test.(test_data, Ref(training_states); params...)
        @info "weights" weights=training_states.parameters
        # if epoch % save_periode == 0
        #     serialize("$(homedir())/$(conf["paths"]["model_dir"])/model_$epoch",
        #         training_states)
        # end
    end
end
function train(data,
        training_states::Lux.Experimental.TrainState;
        params...)
    for d in data
        training_states = train(d, training_states; params...)
        training_states
    end
	training_states
end

function box_coordinate(reduce,fold, collection)
    mapreduce(collect∘reduce, (x, y) -> fold.(x, y), collection)
end
function oct_tree(f::Function, x::AbstractVector, rf::AbstractRefinery)::Cell
    min_coordinate = box_coordinate(f,min,x) |> SVector{3}
    max_coordinate = box_coordinate(f,max,x) |> SVector{3}
    tree = Cell(min_coordinate, max_coordinate, x)
    adaptivesampling!(tree, rf)
    tree
end
function point_grid(atoms::Cell; scale::Float32, r::Float32)::Vector{Point3{Float32}}
    (; origin, widths) = atoms.boundary
    filter(Iterators.product(range.(origin,
        origin .+ widths,
        ; step = scale)...) .|> Point3) do point
        distance2to_center(point, atoms) < r^2
    end
end

"""
    loss_fn(model, ps, st, (; point, atoms, skin))

The loss function used by in training.
compare the predicted (square) distance with \$\\frac{1 + \tanh(d)}{2}\$
Return the error with the espected distance as a metric.
"""
function loss_fn(model, ps, st, (; point, atoms, skin))
    d_pred, st = Lux.apply(model, ModelInput(point, atoms), ps, st)
    d_pred = only(d_pred)
    d_real = distance2(point, skin)
    (d_pred - (1 + tanh(d_real)) / 2)^2,
    st,
    (; distance = abs(d_real - atanh(2d_pred - 1)))
end
function train((; atoms, skin)::TrainingData{Float32},
        training_states::Lux.Experimental.TrainState; scale::Float32,
        r::Float32)
    skin = oct_tree(identity, coordinates(skin), DensityRefinery(10_000, identity))
	atoms = oct_tree(sph -> sph.center,atoms, DensityRefinery(100, sph -> sph.center))
    points = point_grid(atoms; scale, r)

    for point in first(shuffle(MersenneTwister(42), points), 20)
        atoms_neighboord = select_radius(r, point, atoms)
        grads, loss, stats, training_states = Lux.Experimental.compute_gradients(AutoZygote(),
            loss_fn,
            (; point, atoms = atoms_neighboord, skin),
            training_states)
        training_states = Lux.Experimental.apply_gradients(training_states, grads)
        @info "train" loss stats
    end
    training_states
end

function test((; atoms, skin)::TrainingData{Float32},
        training_states::Lux.Experimental.TrainState; scale::Float32,
        r::Float32)
    skin = oct_tree(identity, coordinates(skin), DensityRefinery(10_000, identity))
    atoms = oct_tree(sph -> sph.center, atoms, DensityRefinery(100, sph -> sph.center))
    points = point_grid(atoms; scale, r)

    for point in first(shuffle(MersenneTwister(42), points), 20)
        atoms_neighboord = select_radius(r, point, atoms)
        loss, _, stats = loss_fn(training_states.model, training_states.parameters,
            training_states.states,
            (; point, atoms = atoms_neighboord, skin))
        @info "test" loss stats
    end
end

"""
	load_data(T, name::String)

Load a `TrainingData{T}` from current directory.
You should have a pdb and an off file with name `name` in current directory.
"""
function load_data(T::Type{<:Number}, name::String)
    TrainingData{T}(extract_balls(T, read("$name.pdb", PDB)), load("$name.off"))
end

function train()
    train_data, test_data = splitobs(mapobs(shuffle(MersenneTwister(42),
            conf["protein"]["list"])[1:10]) do name
            load_data(Float32, "$datadir/$name")
        end; at = 0.5)
	tb_logger = TBLogger("$(homedir())/$(conf["paths"]["log_dir"])") 
	with_logger(tb_logger) do
        train((train_data, test_data),
            Lux.Experimental.TrainState(MersenneTwister(42), model,
                Adam(0.01));
            nb_epoch = 10,
            save_periode = 1,
            r = 1.5f0,
            scale = 1.0f0)
    end
end
