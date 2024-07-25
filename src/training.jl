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
using ProgressLogging
using ChainRulesCore

function loggit(x)
    log(x) - log(1 - x)
end

function KL(true_probabilities::AbstractArray{T},
        expected_probabilities::AbstractArray{T}) where {T <: Number}
    epsilon = 1.0f-5
    sum(
        true_probabilities .*
        log.((true_probabilities .+ T(epsilon)) ./ (expected_probabilities .+ T(epsilon))),
        dims = 1)
end

struct BayesianStats
    nb_true_positives::Int
    nb_false_negatives::Int
    nb_true::Int
    nb_false::Int
    function BayesianStats(
            nb_true_positives::Int, nb_false_negatives::Int, nb_true::Int, nb_false::Int)
        @assert nb_true_positives<=nb_true "got $nb_true_positives true positives for $nb_true true values"
        @assert nb_false_negatives<=nb_false "got $nb_false_negatives true negatives for $nb_false true values"
        new(nb_true_positives, nb_false_negatives, nb_true, nb_false)
    end
end
function BayesianStats(real::AbstractVector{Bool}, pred::AbstractVector{Bool})
    nb_true_positives = count(real .&& pred)
    nb_false_negatives = count(.!real .&& pred)
    nb_true = count(real)
    nb_false = count(.!real)
    ignore_derivatives() do
        @debug "statistics" nb_true nb_false nb_true_positives nb_false_negatives
    end
    BayesianStats(nb_true_positives, nb_false_negatives, nb_true, nb_false)
end
function reduce_stats((;
        nb_true_positives, nb_false_negatives, nb_true, nb_false)::BayesianStats)
    ignore_derivatives() do
        @debug "reduce" nb_true nb_false nb_true_positives nb_false_negatives
    end

    false_positive_rate = 1 - nb_true_positives / nb_true
    false_negative_rate = nb_false_negatives / nb_false
    error_rate = max(false_positive_rate, false_negative_rate)
    (; false_positive_rate, false_negative_rate, error_rate)
end

function aggregate(x::AbstractArray{BayesianStats})
    nb_true_positives = sum(getproperty.(x, :nb_true_positives))
    nb_false_negatives = sum(getproperty.(x, :nb_false_negatives))
    nb_true = sum(getproperty.(x, :nb_true))
    nb_false = sum(getproperty.(x, :nb_false))
    BayesianStats(nb_true_positives, nb_false_negatives, nb_true, nb_false) |> reduce_stats
end
aggregate(x::AbstractArray{<:Number}) = mean(x)
function aggregate(x::StructArray)
    Dict(keys(StructArrays.components(x)) .=>
        aggregate.(values(StructArrays.components(x))))
end
function aggregate(w::AbstractDict{<:Any, <:AbstractVector})
    r = Dict(
        keys(w) .=> aggregate.(values(w)))
    r[:global] = aggregate(reduce(vcat, values(w)))
    r
end
function aggregate(x::Any)
    error("not expected $x of type $(typeof(x))")
end

CategoricalMetric = @NamedTuple{
    stats::BayesianStats}
function generate_true_probabilities(d_real::AbstractArray)
    epsilon = 1.0f-5
    is_inside = d_real .> epsilon
    is_outside = d_real .< epsilon
    is_surface = abs.(d_real) .<= epsilon
    probabilities = zeros32(2, length(d_real))
    probabilities[1, :] .= (1 - epsilon) * is_inside + 1 / 2 * is_surface +
                           epsilon * is_outside
    probabilities[2, :] .= (1 - epsilon) * is_outside + 1 / 2 * is_surface +
                           epsilon * is_inside
    probabilities
end
"""
    categorical_loss(model, ps, st, (; point, atoms, d_real))

The loss function used by in training.
Return the KL divergence between true probability and empirical probability
Return the error with the espected distance as a metric.
"""
function categorical_loss(model,
        ps,
        st,
        (; point,
            input,
            d_real)::StructVector{GLobalPreprocessed})::Tuple{
        Float32, Any, CategoricalMetric}
    ret = Lux.apply(model, Batch(input), ps, st)
    v_pred, st = ret
    v_pred = vcat(v_pred, -v_pred)
    v_pred = exp.(v_pred) ./ sum(exp.(v_pred); dims = 1)
    v_pred = cpu_device()(v_pred)
    probabilities = ignore_derivatives() do
        generate_true_probabilities(d_real)
    end
    epsilon = 1.0f-5
    true_vec = Iterators.filter(vec(d_real)) do dist
        abs(dist) > epsilon
    end .> 0
    pred_vec = map(Iterators.filter(zip(
        d_real, vec(v_pred[1, :]))) do (dist, _)
        abs(dist) > epsilon
    end) do (_, pred)
        pred > 0.5f0
    end

    (KL(probabilities, v_pred) |> mean,
        st, (; stats = BayesianStats(true_vec, pred_vec)))
end

ContinousMetric = @NamedTuple{
    stats::BayesianStats,
    bias_error::Float32,
    abs_error::Float32,
    bias_distance::Float32,
    abs_distance::Float32}

"""
    continus_loss(model, ps, st, (; point, atoms, d_real))

The loss function used by in training.
compare the predicted (square) distance with \$\\frac{1 + \tanh(d)}{2}\$
Return the error with the espected distance as a metric.
"""
function continus_loss(model,
        ps,
        st,
        (; point,
            input,
            d_real)::StructVector{GLobalPreprocessed})::Tuple{Float32, Any, ContinousMetric}
    ret = Lux.apply(model, Batch(input), ps, st)
    v_pred, st = ret
    v_pred = cpu_device()(v_pred)
    v_real = σ.(d_real)
    error = v_pred .- v_real
    loss = mean(error .^ 2)
    D_distance = loggit.(max.(0, v_pred) * (1 .- 1.0f-4)) .- d_real

    epsilon = 1.0f-5
    true_vec = Iterators.filter(vec(d_real)) do dist
        abs(dist) > epsilon
    end .> 0.5f0
    pred_vec = map(Iterators.filter(zip(d_real, vec(v_pred))) do (dist, _)
        abs(dist) > epsilon
    end) do (_, pred)
        pred > 0.5f0
    end
    (loss,
        st,
        (; stats = BayesianStats(true_vec, pred_vec),
            bias_error = mean(error),
            abs_error = mean(abs.(error)),
            bias_distance = mean(D_distance),
            abs_distance = abs.(D_distance) |> mean))
end
function get_cutoff_radius(x::Lux.AbstractExplicitLayer)
    get_preprocessing(x).fun.kargs[:cutoff_radius]
end
get_cutoff_radius(x::Lux.StatefulLuxLayer) = get_cutoff_radius(x.model)

"""
    evaluate_model(
        model::Lux.StatefulLuxLayer, x::Point3f, atoms::AnnotedKDTree; cutoff_radius, default_value = -0.0f0)

	evaluate the model on a single point.
	This function handle the logic in case the point is too far from the atoms. In this case default_value is returned and the model is not run.
"""
function evaluate_model(
        model::Lux.StatefulLuxLayer, x::Point3f, atoms::AnnotedKDTree; cutoff_radius, default_value = -0.0f0)
    if distance(x, atoms.tree) >= cutoff_radius
        default_value
    else
        model((x, atoms)) |> cpu_device() |> first
    end
end
function evaluate_model(
        model::Lux.StatefulLuxLayer, x::Batch{Vector{Point3f}}, atoms::AnnotedKDTree;
        cutoff_radius, default_value = -0.0f0)
    is_close = map(x.field) do x
        distance(x, atoms.tree) >= cutoff_radius
    end
    close_points = x.field[is_close] |> Batch
    close_values = model((close_points, atoms)) |> cpu_device() |> first
    ifelse.(is_close, close_values, default_value)
end
"""
    implicit_surface(atoms::AnnotedKDTree{Sphere{T}, :center, Point3{T}},
        model::Lux.StatefulLuxLayer, (;
            cutoff_radius, step)) where {T}

	Create a mesh form the isosurface of function `pos -> model(atoms,pos)` using marching cubes algorithm and using step size `step`.  
"""
function implicit_surface(atoms::AnnotedKDTree{Sphere{T}, :center, Point3{T}},
        model::Lux.StatefulLuxLayer, (;
            cutoff_radius, default_value, iso_value, step)) where {T}
    (; mins, maxes) = atoms.tree.hyper_rec
    ranges = range.(mins, maxes; step)
    grid = Point3f.(reshape(ranges[1], :, 1, 1), reshape(ranges[2], 1, :, 1),
        reshape(ranges[3], 1, 1, :))
    volume = Folds.map(grid) do x
        evaluate_model(model, x, atoms; cutoff_radius, default_value)
    end

    isosurface(volume, MarchingCubes(iso = iso_value),
        SVector{3, Float32}, SVector{3, Int}, mins, maxes - mins)
end

function hausdorff_metric((; atoms, skin)::TreeTrainingData,
        model::StatefulLuxLayer, training_parameters::Training_parameters)
    surface = implicit_surface(atoms, model, training_parameters) |>
              first
    if length(surface) >= 1
        distance(surface, skin.tree)
    else
        Inf32
    end
end

function test_protein(
        data::StructVector{GLobalPreprocessed},
        training_states::Lux.Experimental.TrainState, (; categorical)::Training_parameters)
    loss_vec = Float32[]
    stats_vec = StructVector((categorical ? CategoricalMetric : ContinousMetric)[])
    loss_fn = categorical ? categorical_loss : continus_loss
    for d in BatchView(data; batchsize = 200)
        loss, _, stats = loss_fn(training_states.model, training_states.parameters,
            training_states.states, d)
        loss, stats = (loss, stats) .|> cpu_device()
        push!(loss_vec, loss)
        push!(stats_vec, stats)
    end
    (; loss = loss_vec, stats = stats_vec) |> StructVector
end

function train_protein(
        data::StructVector{GLobalPreprocessed},
        training_states::Lux.Experimental.TrainState, (; categorical)::Training_parameters)
    loss_vec = Float32[]
    stats_vec = StructVector((categorical ? CategoricalMetric : ContinousMetric)[])
    loss_fn = categorical ? categorical_loss : continus_loss
    for d in BatchView(data; batchsize = 200)
        grads, loss, stats, training_states = Lux.Experimental.compute_gradients(
            AutoZygote(),
            loss_fn,
            d,
            training_states)
        @assert !isnan(loss)
        training_states = Lux.Experimental.apply_gradients(training_states, grads)
        loss, stats = (loss, stats) .|> cpu_device()
        push!(loss_vec, loss)
        push!(stats_vec, stats)
    end

    training_states, (; loss = loss_vec, stats = stats_vec) |> StructVector
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

struct DataSet
    outside::StructVector{GLobalPreprocessed}
    surface::StructVector{GLobalPreprocessed}
    inside::StructVector{GLobalPreprocessed}
    core::StructVector{GLobalPreprocessed}
    atoms_center::StructVector{GLobalPreprocessed}
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
    processing = Function[
        (; atoms, skin)::TreeTrainingData -> first(
            approximates_points(
                MersenneTwister(42), atoms.tree, skin.tree, training_parameters) do point
                -2training_parameters.cutoff_radius < signed_distance(point, skin) < 0
            end,
            3000),
        (; atoms, skin)::TreeTrainingData -> first(
            exact_points(
                MersenneTwister(42), atoms.tree, skin.tree, training_parameters),
            3000),
        (; atoms, skin)::TreeTrainingData -> first(
            approximates_points(
                MersenneTwister(42), atoms.tree, skin.tree, training_parameters) do point
                0 < signed_distance(point, skin) < 2training_parameters.cutoff_radius
            end,
            1500),
        (; atoms, skin)::TreeTrainingData -> first(
            approximates_points(
                MersenneTwister(42), atoms.tree, skin.tree, training_parameters) do point
                distance(point, skin.tree) > 2 * training_parameters.cutoff_radius
            end,
            1200),
        (; atoms)::TreeTrainingData -> first(
            shuffle(MersenneTwister(42), atoms.data.center), 300)
    ]
    train_data, test_data = map([train_data, test_data]) do data
        DataSet(Folds.map(processing) do f
            pre_compute_data_set(f, model, data, training_parameters) |> StructVector
        end...)
    end
    @info "end pre computing"

    @info "Starting training"
    @progress name="training" for epoch in 1:nb_epoch
        # for epoch in 1:nb_epoch
        prop = propertynames(train_data)
        train_v = Dict{Symbol, StructVector}()
        for p::Symbol in prop
            training_states, _train_v = train_protein(
                getproperty(train_data, p), training_states, training_parameters)
            train_v[p] = _train_v
        end
        test_v = Dict(
            prop .=>
            test_protein.(getproperty.(Ref(test_data), prop),
                Ref(training_states), Ref(training_parameters)))
        @info "log" test=aggregate(test_v) train=aggregate(train_v)

        if epoch % save_periode == 0
            serialize(
                "$(homedir())/$(model_dir)/$(generate_training_name(training_parameters,epoch))",
                serialized_model_from_preprocessed_states(
                    training_states, training_parameters))
        end
    end
    @info "Stop training"
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
    (; model, learning_rate) = training_parameters
    (; log_dir) = directories
    optim = OptimiserChain(WeightDecay(), Adam(learning_rate))
    (; train_data, test_data) = get_dataset(training_parameters, directories)
    with_logger(get_logger("$(homedir())/$log_dir/$(generate_training_name(training_parameters))")) do
        train((train_data, test_data),
            Lux.Experimental.TrainState(
                MersenneTwister(42), drop_preprocessing(model()), optim) |>
            gpu_device(),
            training_parameters, directories)
    end
end
