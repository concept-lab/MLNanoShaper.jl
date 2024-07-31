using Test

@testset "MLNanoShaper.jl" begin
    @test begin
        using TOML,MLUtils,Optimisers,Logging,TensorBoardLogger,Lux,Random
        using MLNanoShaper
        conf = TOML.parsefile(MLNanoShaper.params_file)
        conf["AuxiliaryParameters"]["nb_epoch"] = 2 |> UInt
		conf["AuxiliaryParameters"]["model_dir"] = mktempdir()
        training_parameters = MLNanoShaper.read_from_TOML(MLNanoShaper.TrainingParameters, conf)
        auxiliary_parameters = MLNanoShaper.read_from_TOML(MLNanoShaper.AuxiliaryParameters, conf)
        train_data, test_data = splitobs(
            mapobs([1, 2]) do id
                MLNanoShaper.load_data_pqr(Float32, "../examples/$id")
            end; at = 0.5)
        (; model, learning_rate) = training_parameters
        log_dir = mktempdir()
        optim = OptimiserChain(WeightDecay(), Adam(learning_rate))
        with_logger(MLNanoShaper.get_logger("$log_dir/$(MLNanoShaper.generate_training_name(training_parameters))")) do
            MLNanoShaper._train((train_data, test_data),
                Lux.Training.TrainState(
                    MersenneTwister(42), MLNanoShaper.drop_preprocessing(model()), optim) |>
                gpu_device(),
                training_parameters, auxiliary_parameters)
        end
		true
    end
end
