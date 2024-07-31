using MLNanoShaper
using Test

@testset "MLNanoShaper.jl" begin
    @test begin
		using TOML
        conf = TOML.parsefile(params_file)
        conf["AuxiliaryParameters"]["nb_epoch"] = 2 |> UInt
        training_parameters = read_from_TOML(TrainingParameters, conf)
        auxiliary_parameters = read_from_TOML(AuxiliaryParameters, conf)
        train_data, test_data = splitobs(
            mapobs([1, 2]) do id
                load_data_pqr(Float32, "$(homedir())/$data_dir/$id")
            end; at = 0.5)
        (; model, learning_rate) = training_parameters
        log_dir = mktempdir()
        optim = OptimiserChain(WeightDecay(), Adam(learning_rate))
        with_logger(get_logger("$(homedir())/$log_dir/$(generate_training_name(training_parameters))")) do
            _train((train_data, test_data),
                Lux.Training.TrainState(
                    MersenneTwister(42), drop_preprocessing(model()), optim) |>
                gpu_device(),
                training_parameters, directories)
        end
    end
end
