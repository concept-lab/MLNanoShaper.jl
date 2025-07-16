### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ eca6b7ec-5ca6-11f0-361c-6da60b9b50fa
using Pkg;Pkg.activate(".")

# ╔═╡ 144b7d4d-e10a-44e0-8a8d-7dc9ea7c56e6
using Revise

# ╔═╡ f321edd2-6ce3-48a7-aa78-fdcba57a7765
using MLNanoShaper, MLNanoShaperRunner, FileIO, StructArrays, Static, Serialization,
      GeometryBasics, LuxCUDA, Lux, Profile, ProfileSVG, ChainRulesCore, Folds,
      BenchmarkTools, Zygote, Distances, LinearAlgebra, LoopVectorization, Folds,
      StaticTools, PProf, CUDA, Adapt, NearestNeighbors, MarchingCubes, FileIO, Random,Transducers,Accessors, TOML, Optimisers, MLCore, PProf

# ╔═╡ 4fef1e98-0aae-4213-b8a4-91465d33bfe0
import CairoMakie as Mk

# ╔═╡ 9a0ee2f4-de3b-4479-9055-06445c7e6cbe
import MLNanoShaper as MLN , MLNanoShaperRunner as MLNR

# ╔═╡ 05e3833a-ccfb-4336-91dc-6cbc1b7572c0
parms = TOML.parsefile(MLN.params_file) 

# ╔═╡ 3f96272e-18ea-401e-aa1a-a6b7f48a4fdb
trp = MLN.read_from_TOML(MLN.TrainingParameters,parms)

# ╔═╡ 833f2cc4-ac6c-422e-928a-bd5128eb3875
auxp = MLN.read_from_TOML(MLN.AuxiliaryParameters,parms)

# ╔═╡ 040ebbdd-872c-4b6d-8037-826b50f39c03
(;test_data) = MLN.get_dataset(trp,auxp)

# ╔═╡ 26846723-d656-41a5-8fd8-c9632713ff75
optim =OptimiserChain(
	ClipGrad(trp.learning_rate/2),
	WeightDecay(),Adam(trp.learning_rate),ClipGrad())

# ╔═╡ 633f421b-05b8-4095-b333-edac2a6efa45
ps,st = (cu(Lux.initialparameters(MersenneTwister(42),trp.model())),cu(Lux.initialstates(MersenneTwister(42), trp.model())))


# ╔═╡ dcb81d07-2db9-4f84-8bae-034d4214bd97

# ╔═╡ ebfd4f24-1def-4b9f-8829-1566154ab1e5
(;atoms_tree, atoms_grid,skin) = MLN.TreeTrainingData(getobs(test_data,1),3f0)

# ╔═╡ aed0581a-2bcb-4e7e-a96f-14bd454efca0
points =  first(MLN.approximates_points(
                MersenneTwister(42), atoms_tree.tree, skin.tree, trp) do point
                true
            end,100_000)

# ╔═╡ f87035b3-3351-40eb-8491-fd893939ad1e
inputs = get_preprocessing(trp.model())((Batch(points),atoms_grid))
# ╔═╡ 55236a97-1649-4c67-971c-8b232ab18639
d_reals = MLN.signed_distance.(points,Ref(skin))

# ╔═╡ 6fe25579-93db-439e-bf46-6ab901211c1f
begin
	Profile.clear()
	f(x,_,_) = zeros(Float32,length(x.lengths) -1),(;)
	MLN.continuous_loss(f,ps,st,(;inputs,d_reals))
 	@pprof [ MLN.continuous_loss(f,ps,st,(;inputs,d_reals)) for _ in 1:1000] 
 	while true
		pprof()
	end
end
# ╔═╡ Cell order:
# ╠═eca6b7ec-5ca6-11f0-361c-6da60b9b50fa
# ╠═144b7d4d-e10a-44e0-8a8d-7dc9ea7c56e6
# ╠═4fef1e98-0aae-4213-b8a4-91465d33bfe0
# ╠═f321edd2-6ce3-48a7-aa78-fdcba57a7765
# ╠═9a0ee2f4-de3b-4479-9055-06445c7e6cbe
# ╠═05e3833a-ccfb-4336-91dc-6cbc1b7572c0
# ╠═3f96272e-18ea-401e-aa1a-a6b7f48a4fdb
# ╠═833f2cc4-ac6c-422e-928a-bd5128eb3875
# ╠═040ebbdd-872c-4b6d-8037-826b50f39c03
# ╠═26846723-d656-41a5-8fd8-c9632713ff75
# ╠═633f421b-05b8-4095-b333-edac2a6efa45
# ╠═ebfd4f24-1def-4b9f-8829-1566154ab1e5
# ╠═aed0581a-2bcb-4e7e-a96f-14bd454efca0
# ╠═f87035b3-3351-40eb-8491-fd893939ad1e
# ╠═55236a97-1649-4c67-971c-8b232ab18639
# ╠═6fe25579-93db-439e-bf46-6ab901211c1f
