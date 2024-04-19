using MLNanoShaper
using Test


@testset "MLNanoShaper.jl" begin
	@test begin 
		using Random, MLNanoShaper, Lux,BioStructures,GeometryBasics
		prot = read("../examples/1MH1.pdb",PDB)
		balls = extract_balls(prot)
		ps = Lux.initialparameters(MersenneTwister(42), MLNanoShaper.model)
		length(LuxCore.stateless_apply(MLNanoShaper.model,MLNanoShaper.ModelInput(first(balls).center,balls),ps)) ==1
	end
	@test begin
	end
end

 

 
