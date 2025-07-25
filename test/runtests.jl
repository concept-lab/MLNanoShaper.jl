using Test
using TOML, MLUtils, Optimisers, Logging, TensorBoardLogger, Lux, Random, Zygote, Accessors 
using MLNanoShaper, MLNanoShaperRunner
Random.seed!(1234)
conf = TOML.parsefile(MLNanoShaper.params_file)
conf["AuxiliaryParameters"]["nb_epoch"] = 1 |> UInt
conf["AuxiliaryParameters"]["model_dir"] = mktempdir()
conf["TrainingParameters"]["loss"] = :categorical
training_parameters = MLNanoShaper.read_from_TOML(
    MLNanoShaper.TrainingParameters, conf)
auxiliary_parameters = MLNanoShaper.read_from_TOML(
    MLNanoShaper.AuxiliaryParameters, conf)
conf["TrainingParameters"]["loss"] = :continuous
training_parameters_c = MLNanoShaper.read_from_TOML(
    MLNanoShaper.TrainingParameters, conf)
(; model, learning_rate) = training_parameters
log_dir = mktempdir()
optim = OptimiserChain(ClipGrad(),WeightDecay(), Adam(learning_rate))
ps = Lux.initialparameters(MersenneTwister(42), model())
st = Lux.initialstates(MersenneTwister(42), model())

@testset "training" begin
    @test begin
        train_data, test_data = splitobs(
            mapobs([1, 2]) do id
                MLNanoShaper.load_data_pqr(
                    Float32, "$(dirname(dirname(@__FILE__)))/examples/$id")
            end; at = 0.5)
        with_logger(MLNanoShaper.get_logger("$log_dir/$(MLNanoShaper.generate_training_name(training_parameters))",0)) do
            MLNanoShaper._train((train_data, test_data),
                Lux.Training.TrainState(
                    MLNanoShaper.drop_preprocessing(model(on_gpu=false)),
                    ps,
                    st,
                    optim),
                training_parameters, auxiliary_parameters)
        end
        true
    end
    @test begin
        train_data, test_data = splitobs(
            mapobs([1, 2]) do id
                MLNanoShaper.load_data_pqr(
                    Float32, "$(dirname(dirname(@__FILE__)))/examples/$id")
            end; at = 0.5)
        with_logger(MLNanoShaper.get_logger("$log_dir/$(MLNanoShaper.generate_training_name(training_parameters))",0)) do
            MLNanoShaper._train((train_data, test_data),
                Lux.Training.TrainState(
                    MLNanoShaper.drop_preprocessing(model(on_gpu=false)),
                    ps,
                    st,
                    optim),
                training_parameters_c ,auxiliary_parameters)
        end
        true
    end
end
