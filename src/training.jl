using Base: fieldindex
using MLNanoShaperRunner: stack_ConcatenatedBatch
using Optimisers: trainable
function get_cutoff_radius(x::Lux.AbstractLuxLayer)
   get_preprocessing(x).fun.kargs[:cutoff_radius]
end
get_cutoff_radius(x::Lux.StatefulLuxLayer) = get_cutoff_radius(x.model)

"""
    implicit_surface(atoms::RegularGrid{T},
        model::Lux.StatefulLuxLayer, (;
            cutoff_radius, step)) where {T}

	Create a mesh form the isosurface of function `pos -> model(atoms,pos)` using marching cubes algorithm and using step size `step`.  
"""
function implicit_surface(model::Lux.StatefulLuxLayer,
    atoms::RegularGrid{T};
            iso_value=.5, step=.5)::ConcatenatedBatch where {T}
    (; mins, maxes) = atoms.tree.hyper_rec
    ranges = range.(mins, maxes; step)
    grid = Point3f.(reshape(ranges[1], :, 1, 1), reshape(ranges[2], 1, :, 1),
        reshape(ranges[3], 1, 1, :))
    @info "computing volume" nb_points=length(grid)
    volume = Folds.map(grid) do x
        model((x, atoms))
    end

    @info "computing isosurface"
    isosurface(volume, MarchingCubes(iso = iso_value),
        SVector{3, Float32}, SVector{3, Int}, mins, maxes - mins)
end
function batch_dataset((; inputs, d_reals)::GlobalPreprocessed)
    mapobs(1:(length(inputs))) do i
        let inputs = get_element(inputs, i),
            d_reals = d_reals[i]

            (; inputs, d_reals)::GlobalPreprocessed
        end
    end
end
function test_protein(
        data::GlobalPreprocessed,
        training_states::Lux.Training.TrainState, (; loss)::TrainingParameters,(;batch_size)::AuxiliaryParameters)
    loss_vec = Float32[]
    stats_vec = StructVector((metric_type(loss))[])
    loss_fn = get_loss_fn(loss)
    # data = batch_dataset(data)
    for data_batch in BatchView(data; batchsize = batch_size)
        loss, _, stats = loss_fn(training_states.model, training_states.parameters,
            Lux.testmode(training_states.states), data_batch)
        loss, stats = (loss, stats) .|> cpu_device()
        push!(loss_vec, loss)
        push!(stats_vec, stats)
    end
    (; loss = loss_vec, stats = stats_vec) |> StructVector
end

function train_protein(
        data,
        training_states::Lux.Training.TrainState, (; loss)::TrainingParameters,(;batch_size)::AuxiliaryParameters)
    loss_vec = Float32[]
    stats_vec = StructVector((metric_type(loss))[])
    loss_fn = get_loss_fn(loss)
    # data = batch_dataset(data)
    # @info "batch" data
    for data_batch in BatchView(data; batchsize = batch_size)
       # @info "batch length" length(data_batch.inputs) Base.summarysize(data_batch)/1024^3 
        grads, loss, stats, training_states = Lux.Training.compute_gradients(
            AutoZygote(),
            loss_fn,
            data_batch,
            training_states)
        # @info loss
        @assert !isnan(loss)
        training_states = Lux.Training.apply_gradients(training_states, grads)
        loss, stats = (loss, stats) .|> cpu_device()
        push!(loss_vec, loss)
        push!(stats_vec, stats)
    end

    training_states, (; loss = loss_vec, stats = stats_vec) |> StructVector
end

function serialized_model_from_preprocessed_states(
        (; parameters, states)::Lux.Training.TrainState, y::TrainingParameters)
    MLNanoShaperRunner.SerializedModel(
        y.model, parameters |> cpu_device(), states |> cpu_device())
end

struct DataSet
    outside::GlobalPreprocessed
    surface::GlobalPreprocessed
    inside::GlobalPreprocessed
    core::GlobalPreprocessed
    # atoms_center::GlobalPreprocessed
end

function join_dataset(vec)
    inputs = stack_ConcatenatedBatch(getproperty.(vec,:inputs) |> collect)
    d_reals =  vcat(getproperty.(vec,:d_reals)...)
    (;inputs,d_reals)
end
function shuffle_dataset(rng::AbstractRNG,vec::GlobalPreprocessed)
    perm = randperm(rng,length(vec.d_reals))
    (; inputs = get_element(vec.inputs, perm), d_reals = vec.d_reals[perm])
end
"""
	train((train_data,test_data),training_states; nb_epoch)
train the model on the data with nb_epoch
"""

function _train(
        (train_data,
            test_data)::Tuple{MLUtils.AbstractDataContainer, MLUtils.AbstractDataContainer},
        training_states::Lux.Training.TrainState, training_parameters::TrainingParameters,
        auxiliary_parameters::AuxiliaryParameters)
    (; nb_epoch, save_periode, model_dir,batch_size) = auxiliary_parameters

    @info "nb threades" Threads.nthreads()
    @info "building KDtrees"
    cutoff_radius = training_parameters.cutoff_radius
    train_data = Folds.map(x -> TreeTrainingData(x,cutoff_radius), train_data)
    test_data  = Folds.map(x -> TreeTrainingData(x,cutoff_radius), test_data )

    @info "pre computing"
    model = get_preprocessing(training_parameters.model())
    processing = Function[
        (; atoms_tree, skin)::TreeTrainingData -> first(
            approximates_points(
                MersenneTwister(42), atoms_tree.tree, skin.tree, training_parameters) do point
                -training_parameters.cutoff_radius < signed_distance(point, skin) < 0
            end,
            1000),
        (; atoms_tree, skin)::TreeTrainingData -> first(
            exact_points(
                MersenneTwister(42), atoms_tree.tree, skin.tree, training_parameters),
            500),
        (; atoms_tree, skin)::TreeTrainingData -> first(
            approximates_points(
                MersenneTwister(42), atoms_tree.tree, skin.tree, training_parameters) do point
                0 < signed_distance(point, skin) < training_parameters.cutoff_radius
            end,
            2000),
        (; atoms_tree, skin)::TreeTrainingData -> first(
            approximates_points(
                MersenneTwister(42), atoms_tree.tree, skin.tree, training_parameters) do point
                signed_distance(point, skin) > training_parameters.cutoff_radius
            end,
            1000),
        # (; atoms_tree)::TreeTrainingData -> first(
            # shuffle(MersenneTwister(42), atoms_tree.data.center), 20)
    ]
    train_data, test_data = map([train_data, test_data]) do dataset
        DataSet(map(processing) do generate_points
            pre_compute_data_set(
                generate_points,
                model,
                dataset,
                training_parameters)
        end...)
    end
    @info "end pre computing"
    @info("train data size",
        outside=length(first(train_data.outside)),
        surface=length(first(train_data.surface)),
        inside=length(first(train_data.inside)),
        core=length(first(train_data.core)),
        # atoms_center=length(first(train_data.atoms_center))
        )
    @info("test data size",
        outside=length(first(test_data.outside)),
        surface=length(first(test_data.surface)),
        inside=length(first(test_data.inside)),
        core=length(first(test_data.core)),
        # atoms_center=length(first(test_data.atoms_center))
        )
    train_data = shuffle_dataset(MersenneTwister(41), join_dataset(getproperty.(Ref(train_data),propertynames(train_data))))
    @info "training size: $(floor((Base.summarysize(train_data) + Base.summarysize(test_data))/1024^3;digits=3)) Go"
    @info "example batch size: $(floor(mean(Base.summarysize.(BatchView(train_data;batchsize =batch_size)))/1024^3;digits=3)) Go"

    @info "Starting training"
    η = training_parameters.learning_rate
    @progress name="training" for epoch in 1:nb_epoch
        # for epoch in 1:nb_epoch
        #train
        training_states, train_v = train_protein(train_data, training_states, training_parameters,auxiliary_parameters)
        loss_mean = train_v.loss |> mean
        counter = 1 + (loss_mean > .6) + (loss_mean > .5) +  (loss_mean > .4) + (loss_mean > .35)
        η = (1e-5,5e-5,1e-4,2e-4,5e-4)[counter]
        Optimisers.adjust!(training_states.optimizer_state,η,)
        Optimisers.adjust!(training_states.optimizer_state,lambda = η)
        #test
        prop = propertynames(test_data)
        test_v = Dict(
            prop .=>
            test_protein.(getproperty.(Ref(test_data), prop),
                Ref(training_states), Ref(training_parameters),Ref(auxiliary_parameters)))
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
    _train(training_parameters::TrainingParameters, directories::AuxiliaryParameters)

train the model given `TrainingParameters` and `AuxiliaryParameters`.
"""
function _train(training_parameters::TrainingParameters, auxiliary_parameters::AuxiliaryParameters)
    (; model, learning_rate) = training_parameters
    (; log_dir, on_gpu) = auxiliary_parameters
    device = on_gpu ? gpu_device() : identity
    optim = OptimiserChain(ClipGrad(learning_rate/2),WeightDecay(),Adam(learning_rate),ClipGrad())
    (; train_data, test_data) = get_dataset(training_parameters, auxiliary_parameters)
    ps = Lux.initialparameters(MersenneTwister(42), model())
    st = Lux.initialstates(MersenneTwister(42), model())
    with_logger(get_logger("$(homedir())/$log_dir/$(generate_training_name(training_parameters))")) do
        _train((train_data, test_data),
            Lux.Training.TrainState(
                drop_preprocessing(model()),
                ps |> device,
                st |> device,
                optim),
            training_parameters, auxiliary_parameters)
    end
end
