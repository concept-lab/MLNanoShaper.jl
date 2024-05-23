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
using Folds
using Static

function implicit_surface(atoms_tree::KDTree, atoms::StructVector{Sphere{Float32}},
        training_states::Lux.Experimental.TrainState, (;
            cutoff_radius)::Training_parameters)
    (; mins, maxes) = atoms_tree.hyper_rec
    isosurface(
        MarchingCubes(), SVector{3, Float32}; origin = mins, widths = maxes - mins) do x
        atoms_neighboord = atoms[inrange(atoms_tree, x, cutoff_radius)] |> StructVector
        if length(atoms_neighboord) > 0
            training_states.model(ModelInput(Point3f(x), atoms_neighboord),
                training_states.parameters, training_states.states) |> first
        else
            0.0f0
        end - 0.5f0
    end
end

"""
    loss_fn(model, ps, st, (; point, atoms, d_real))

The loss function used by in training.
compare the predicted (square) distance with \$\\frac{1 + \tanh(d)}{2}\$
Return the error with the espected distance as a metric.
"""
function loss_fn(model,
        ps,
        st,
        (; point,
            atoms,
            d_real)::StructVector{@NamedTuple{
            point::Point3f, atoms::StructVector{Sphere{Float32}}, d_real::Float32}})
    ret = Lux.apply(model, Batch(ModelInput.(point, atoms)), ps, st)
    v_pred, st = ret
	v_pred = cpu_device()(v_pred)

    ((v_pred .- (1 .+ tanh.(d_real)) ./ 2) .^ 2 |> mean,
        st,
        (;
            distance = abs.(d_real .- atanh.(max.(0, (2v_pred .- 1)) * (1 .- 1.0f-4))) |>
                       mean))
end

function hausdorff_metric((; atoms, atoms_tree, skin)::TreeTrainingData,
        training_states::Lux.Experimental.TrainState, training_parameters::Training_parameters)
    surface = implicit_surface(atoms_tree, atoms, training_states, training_parameters) |>
              first
    if length(surface) >= 1
        distance(surface, skin.tree)
    else
        Inf32
    end
end

function test(
        data::StructVector{@NamedTuple{
            point::Point3f, atoms::StructVector{Sphere{Float32}}, d_real::Float32}},
        training_states::Lux.Experimental.TrainState)
    loss_vec = Float32[]
    stats_vec = StructVector(@NamedTuple{distance::Float32}[])
    for d in BatchView(data; batchsize = 200)
        loss, _, stats = loss_fn(training_states.model, training_states.parameters,
            training_states.states, d)
        loss, stats = (loss, stats) .|> cpu_device()
        push!(loss_vec, loss)
        push!(stats_vec, stats)
    end
    loss, distance = mean(loss_vec), mean(stats_vec.distance)
    (; loss, distance)
end

function train(
        data::StructVector{@NamedTuple{
            point::Point3f, atoms::StructVector{Sphere{Float32}}, d_real::Float32}},
        training_states::Lux.Experimental.TrainState)
    loss_vec = Float32[]
    stats_vec = StructVector(@NamedTuple{distance::Float32}[])
    for d in BatchView(data; batchsize = 200)
        grads, loss, stats, training_states = Lux.Experimental.compute_gradients(
            AutoZygote(),
            loss_fn,
            d,
            training_states)
        training_states = Lux.Experimental.apply_gradients(training_states, grads)
        loss, stats = (loss, stats) .|> cpu_device()
        push!(loss_vec, loss)
        push!(stats_vec, stats)
    end
    loss, distance = mean(loss_vec), mean(stats_vec.distance)
    training_states, (; loss, distance)
end

"""
	train((train_data,test_data),training_states; nb_epoch)
train the model on the data with nb_epoch
"""
function train(
        (train_data,
            test_data)::Tuple{MLUtils.AbstractDataContainer, MLUtils.AbstractDataContainer},
        training_states::Lux.Experimental.TrainState, training_parameters::Training_parameters,
        auxiliary_parameters::Auxiliary_parameters)
    (; nb_epoch, save_periode, model_dir) = auxiliary_parameters

    @info "building KDtrees"
    train_data = Folds.map(TreeTrainingData, train_data)
    test_data = Folds.map(TreeTrainingData, test_data)
    @info "pre computing"

    train_data = pre_compute_data_set(
        train_data, training_parameters) do (; atoms, skin), tr
        vcat(first(point_grid(MersenneTwister(42), atoms.tree, skin.tree, tr),
                40),
            first(exact_points(MersenneTwister(42), atoms.tree, skin.tree, tr), 40))
    end |> StructVector
    test_data_approximate = pre_compute_data_set(
        test_data, training_parameters) do (; atoms, skin), tr
        first(point_grid(MersenneTwister(42), atoms.tree, skin.tree, tr), 40)
    end |> StructVector
    test_data_exact = pre_compute_data_set(
        test_data, training_parameters) do (; atoms, skin), tr
        first(exact_points(MersenneTwister(42), atoms.tree, skin.tree, tr), 40)
    end |> StructVector
    @info "end pre computing"

    for epoch in 1:nb_epoch
        @info "epoch" epoch=Int(epoch)
        test_exact = test(test_data_exact, training_states)
        test_approximate = test(test_data_approximate, training_states)
        training_states, train_v = train(train_data, training_states)
        @info "test" exact=test_exact approximate=test_approximate
        @info "train" loss=train_v.loss distance=train_v.distance

        if epoch % save_periode == 0
            serialize(
                "$(homedir())/$(model_dir)/$(generate_training_name(training_parameters,epoch))",
				training_states.parameters |>cpu_device())
        end
    end
end

"""
    train(training_parameters::Training_parameters, directories::Auxiliary_parameters)

train the model given `Training_parameters` and `Auxiliary_parameters`.
"""
function train(training_parameters::Training_parameters, directories::Auxiliary_parameters)
    (; data_ids, train_test_split, model) = training_parameters
    (; data_dir, log_dir) = directories
    train_data, test_data = splitobs(
        mapobs(shuffle(MersenneTwister(42),
            data_ids)) do id
            load_data_pqr(Float32, "$(homedir())/$data_dir/$id")
        end; at = train_test_split)
    optim = OptimiserChain(WeightDecay(), Adam())
    with_logger(get_logger("$(homedir())/$log_dir/$(generate_training_name(training_parameters))")) do
        train((train_data, test_data),
            Lux.Experimental.TrainState(MersenneTwister(42), model, optim) |> gpu_device(),
            training_parameters, directories)
    end
end
