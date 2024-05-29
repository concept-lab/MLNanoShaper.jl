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
            input,
            d_real)::StructVector{GLobalPreprocessed})
    ret = Lux.apply(model, Batch(input), ps, st)
    v_pred, st = ret
    v_pred = cpu_device()(v_pred)

    ((v_pred .- (1 .+ tanh.(d_real)) ./ 2) .^ 2 |> mean,
        st,
        (;
            distance = abs.(d_real .- atanh.(max.(0, (2v_pred .- 1)) * (1 .- 1.0f-4))) |>
                       mean))
end
get_cutoff_radius(x::Lux.AbstractExplicitLayer) = get_preprocessing(x).fun.kargs[:cutoff_radius]
get_cutoff_radius(x::Lux.StatefulLuxLayer) = get_cutoff_radius(x.model)
function implicit_surface(atoms::AnnotedKDTree{Sphere{T}, :center, Point3{T}},
        model::Lux.StatefulLuxLayer, (;
            cutoff_radius)::Training_parameters) where {T}
    (; mins, maxes) = atoms.tree.hyper_rec
    cutoff_radius = get_cutoff_radius(model)
    isosurface(
        MarchingCubes(), SVector{3, Float32}; origin = mins, widths = maxes - mins) do x
        if distance(Point3f(x), atoms.tree) >= cutoff_radius
            0.0f0
        else
			only(model((Point3f(x), atoms)))-0.5f0
        end
    end
end

function hausdorff_metric((; atoms, skin)::TreeTrainingData,
        model::StatefulLuxLayer, training_parameters::Training_parameters)
    surface = implicit_surface(atoms, model, training_parameters) |>
              first
    if length(surface) >= 0
        distance(surface, skin.tree)
    else
        Inf32
    end
end

function test(
        data::StructVector{GLobalPreprocessed},
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
        data::StructVector{GLobalPreprocessed},
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

function serialized_model_from_preprocessed_states(
        (; parameters)::Lux.Experimental.TrainState, y::Training_parameters)
    parameters = [Symbol("layer_$i") => if i == 1
                      (;)
                  else
                      parameters[keys(parameters)[i - 1]]
                  end for i in 1:(1 + length(keys(parameters)))] |> NamedTuple
    SerializedModel(y.model, parameters |> cpu_device())
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
    model = get_preprocessing(training_parameters.model())

    train_data = pre_compute_data_set(
        model, train_data) do (; atoms, skin)
        vcat(
            first(
                point_grid(MersenneTwister(42), atoms.tree, skin.tree, training_parameters),
                40),
            first(
                exact_points(
                    MersenneTwister(42), atoms.tree, skin.tree, training_parameters),
                40))
    end |> StructVector
    test_data_approximate = pre_compute_data_set(
        model, test_data) do (; atoms, skin)
        first(point_grid(MersenneTwister(42), atoms.tree, skin.tree, training_parameters),
            40)
    end |> StructVector
    test_data_exact = pre_compute_data_set(
        model, test_data) do (; atoms, skin)
        first(
            exact_points(MersenneTwister(42), atoms.tree, skin.tree, training_parameters),
            40)
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
                serialized_model_from_preprocessed_states(
                    training_states, training_parameters))
        end
    end
end

function get_dataset((; data_ids, train_test_split)::Training_parameters,
        (; data_dir)::Auxiliary_parameters)
    train_data, test_data = splitobs(
        mapobs(shuffle(MersenneTwister(42),
            data_ids)) do id
            load_data_pqr(Float32, "$(homedir())/$data_dir/$id")
        end; at = train_test_split)
    (; train_data, test_data)
end

"""
    train(training_parameters::Training_parameters, directories::Auxiliary_parameters)

train the model given `Training_parameters` and `Auxiliary_parameters`.
"""
function train(training_parameters::Training_parameters, directories::Auxiliary_parameters)
    (; model) = training_parameters
    (; log_dir) = directories
    optim = OptimiserChain(WeightDecay(), Adam())
    (; train_data, test_data) = get_dataset(training_parameters, directories)
    with_logger(get_logger("$(homedir())/$log_dir/$(generate_training_name(training_parameters))")) do
        train((train_data, test_data),
            Lux.Experimental.TrainState(
                MersenneTwister(42), drop_preprocessing(model()), optim) |>
            gpu_device(),
            training_parameters, directories)
    end
end
