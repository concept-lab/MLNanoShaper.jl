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
        distance(x, atoms.tree) <= cutoff_radius
    end
    close_points = x.field[is_close] |> Batch
    if length(close_points.field) > 0
        close_values = model((close_points, atoms)) |> cpu_device() |> first
        ifelse.(is_close, close_values, default_value)
    else
        zeros(x.field)
    end
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
function batch_dataset((; points, inputs, d_reals)::GlobalPreprocessed)
    mapobs(1:(length(inputs.lengths) - 1)) do i
        let inputs = get_element(inputs, i),
            d_reals = d_reals[i]

            (; inputs, d_reals)::GlobalPreprocessed
        end
    end
end
function test_protein(
        data::GlobalPreprocessed,
        training_states::Lux.Training.TrainState, (; loss)::TrainingParameters)
    loss_vec = Float32[]
    stats_vec = StructVector((metric_type(loss))[])
    loss_fn = get_loss_fn(loss)
    data = batch_dataset(data)
    for data_batch in BatchView(data; batchsize = 2000)
        loss, _, stats = loss_fn(training_states.model, training_states.parameters,
            training_states.states, data_batch)
        loss, stats = (loss, stats) .|> cpu_device()
        push!(loss_vec, loss)
        push!(stats_vec, stats)
    end
    (; loss = loss_vec, stats = stats_vec) |> StructVector
end

function train_protein(
        data::GlobalPreprocessed,
        training_states::Lux.Training.TrainState, (; loss)::TrainingParameters)
    loss_vec = Float32[]
    stats_vec = StructVector((metric_type(loss))[])
    loss_fn = get_loss_fn(loss)
    data = batch_dataset(data)
    for data_batch in BatchView(data; batchsize = 2000)
        grads, loss, stats, training_states = Lux.Training.compute_gradients(
            AutoZygote(),
            loss_fn,
            data_batch,
            training_states)
        @assert !isnan(loss)
        training_states = Lux.Training.apply_gradients(training_states, grads)
        loss, stats = (loss, stats) .|> cpu_device()
        push!(loss_vec, loss)
        push!(stats_vec, stats)
    end

    training_states, (; loss = loss_vec, stats = stats_vec) |> StructVector
end

function serialized_model_from_preprocessed_states(
        (; parameters)::Lux.Training.TrainState, y::TrainingParameters)
    parameters = [Symbol("layer_$i") => if i == 1
                      (;)
                  else
                      parameters[keys(parameters)[i - 1]]
                  end for i in 1:(1 + length(keys(parameters)))] |> NamedTuple
    SerializedModel(y.model, parameters |> cpu_device())
end

struct DataSet
    outside::GlobalPreprocessed
    surface::GlobalPreprocessed
    inside::GlobalPreprocessed
    core::GlobalPreprocessed
    atoms_center::GlobalPreprocessed
end

"""
	train((train_data,test_data),training_states; nb_epoch)
train the model on the data with nb_epoch
"""
function train(
        (train_data,
            test_data)::Tuple{MLUtils.AbstractDataContainer, MLUtils.AbstractDataContainer},
        training_states::Lux.Training.TrainState, training_parameters::TrainingParameters,
        auxiliary_parameters::AuxiliaryParameters)
    (; nb_epoch, save_periode, model_dir) = auxiliary_parameters

    @info "nb threades" Threads.nthreads()

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
    train_data, test_data = map([train_data, test_data]) do dataset
        DataSet(Folds.map(processing) do generate_points
            pre_compute_data_set(
                generate_points,
                model,
                dataset,
                training_parameters)
        end...)
    end
    @info "end pre computing"

    @info "Starting training"
    @progress name="training" for epoch in 1:nb_epoch
        # for epoch in 1:nb_epoch
        prop = propertynames(train_data)
        #train
        train_v = Dict{Symbol, StructVector}()
        for p::Symbol in prop
            training_states, _train_v = train_protein(
                getproperty(train_data, p), training_states, training_parameters)
            train_v[p] = _train_v
        end
        #test
        test_v = Dict(
            prop .=>
            test_protein.(getproperty.(Ref(test_data), prop),
                Ref(training_states), Ref(training_parameters)))
        #log
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

function get_dataset((; data_ids, train_test_split)::TrainingParameters,
        (; data_dir)::AuxiliaryParameters)
    train_data, test_data = splitobs(
        mapobs(shuffle(MersenneTwister(42),
            data_ids)) do id
            load_data_pqr(Float32, "$(homedir())/$data_dir/$id")
        end; at = train_test_split)
    (; train_data, test_data)
end

"""
    train(training_parameters::TrainingParameters, directories::AuxiliaryParameters)

train the model given `TrainingParameters` and `AuxiliaryParameters`.
"""
function train(training_parameters::TrainingParameters, directories::AuxiliaryParameters)
    (; model, learning_rate) = training_parameters
    (; log_dir) = directories
    optim = OptimiserChain(WeightDecay(), Adam(learning_rate))
    (; train_data, test_data) = get_dataset(training_parameters, directories)
    with_logger(get_logger("$(homedir())/$log_dir/$(generate_training_name(training_parameters))")) do
        train((train_data, test_data),
            Lux.Training.TrainState(
                MersenneTwister(42), drop_preprocessing(model()), optim) |>
            gpu_device(),
            training_parameters, directories)
    end
end
