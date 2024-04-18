using MLNanoShaper
using Test


@testset "MLNanoShaper.jl" begin
	@test begin 
		using Random, MLNanoShaper, Lux,BioStructures,GeometryBasics
		prot = read("../examples/1MH1.pdb",PDB)
		dict = read("../param/protein.r.dict",MLNanoShaper.DICT{Float64})
		balls = extract_balls(prot,dict)
		ps = Lux.initialparameters(MersenneTwister(42), MLNanoShaper.model)
		length(LuxCore.stateless_apply(MLNanoShaper.model,MLNanoShaper.Input(first(balls).center,balls),ps)) ==1
	end
end

 

 
