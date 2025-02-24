using Test

@testset "MLNanoShaper.jl" begin
    @test begin
        using TOML, MLUtils, Optimisers, Logging, TensorBoardLogger, Lux, Random
        using MLNanoShaper
        conf = TOML.parsefile(MLNanoShaper.params_file)
        conf["AuxiliaryParameters"]["nb_epoch"] = 2 |> UInt
        conf["AuxiliaryParameters"]["model_dir"] = mktempdir()
        training_parameters = MLNanoShaper.read_from_TOML(
            MLNanoShaper.TrainingParameters, conf)
        auxiliary_parameters = MLNanoShaper.read_from_TOML(
            MLNanoShaper.AuxiliaryParameters, conf)
        train_data, test_data = splitobs(
            mapobs([1, 2]) do id
                MLNanoShaper.load_data_pqr(
                    Float32, "$(dirname(dirname(@__FILE__)))/examples/$id")
            end; at = 0.5)
        (; model, learning_rate) = training_parameters
        log_dir = mktempdir()
        optim = OptimiserChain(WeightDecay(), Adam(learning_rate))
        ps = Lux.initialparameters(MersenneTwister(42), model())
        st = Lux.initialstates(MersenneTwister(42), model())
        with_logger(MLNanoShaper.get_logger("$log_dir/$(MLNanoShaper.generate_training_name(training_parameters))")) do
            MLNanoShaper._train((train_data, test_data),
                Lux.Training.TrainState(
                    MLNanoShaper.drop_preprocessing(model()),
                    ps |> gpu_device(),
                    st |> gpu_device(),
                    optim),
                training_parameters, auxiliary_parameters)
        end
        true
    end
end
