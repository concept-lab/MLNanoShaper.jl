### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 9e80b73c-92ac-4583-b6ae-bfc94f3ea429
using Pkg; Pkg.activate(".")

# ╔═╡ 53dd5feb-0836-4aff-9c8e-67922cfc548e
using Revise, MLNanoShaper, MLNanoShaperRunner, FileIO, StructArrays, Static, Serialization,
      GeometryBasics, LuxCUDA, Lux, Profile, ProfileSVG, ChainRulesCore, Folds,
      BenchmarkTools, Zygote, Distances, LinearAlgebra, LoopVectorization, Folds,
      StaticTools, PProf, CUDA, Adapt, NearestNeighbors, MarchingCubes, FileIO, Transducers,Accessors, Reactant, Random

# ╔═╡ 6020c872-058e-11f0-325a-53e3477ce60c
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 10%);
	}
</style>
"""

# ╔═╡ 74e95b14-7440-4301-b3ec-4578eaab5cf2
dev = reactant_device()

# ╔═╡ e6a28bfa-353b-4985-882b-97357f1ef46d
model_weights = deserialize("$(homedir())/datasets/models/tiny_angular_dense_s_jobs_14_6_3_c_2025-03-19_epoch_400_9592899277305186470")

# ╔═╡ 2b427a00-c730-4632-bb12-b8c4049d2583
model_fixed_size = Lux.StatefulLuxLayer{true}(model_weights.model(max_nb_atoms = 10),model_weights.parameters,model_weights.states)

# ╔═╡ 15ecb8ac-f4a8-4331-9cde-829f60be5f26
model =  Lux.StatefulLuxLayer{true}(model_weights.model(),model_weights.parameters,model_weights.states)

# ╔═╡ af81d80c-d75b-42ef-84a8-b59ae6d99cff
prot_num = 1

# ╔═╡ c081367e-219f-4092-82c2-c78c99622986
atoms = RegularGrid(
    getfield.(
        read("$(homedir())/datasets/pqr/$prot_num/structure.pqr", PQR{Float32}), :pos) |>
    StructVector,3f0)

# ╔═╡ 45a8d4da-9701-4568-a2d6-64eff31baee3
model_fixed_size((Batch([Point3f(10,6+i,0) for i in 1:10]),atoms))

# ╔═╡ 5a1d21a3-835f-4f2c-8326-a12c174f7392
model((Batch([Point3f(10,6+i,0) for i in 1:10]),atoms))

# ╔═╡ 227239ed-ef7d-417c-925d-faab55c09c95
get_preprocessing(model_weights.model()).fun

# ╔═╡ d9823229-89d4-4310-baeb-74894de7efac
ps = dev(model_weights.parameters)

# ╔═╡ 27d4d940-6842-400d-b496-102be93f79c7
st = dev(model_weights.states)

# ╔═╡ 7650e69e-f022-45f3-9ac1-92602cd0308a
raw_model = drop_preprocessing(model_weights.model(max_nb_atoms = 10))

# ╔═╡ 8ec24e83-2a87-4040-9430-a79b371331c7
dumy_input = randn(Xoshiro(42), Float32, 6, 55, 10) |> dev;

# ╔═╡ b369794d-6b43-4c86-bbf7-42f7d35d6919
code = @code_hlo Lux.apply(raw_model,dumy_input,ps,st)

# ╔═╡ b3a4a3a2-f7eb-4d6c-8718-35c2d36658df
open("exported_lux_model.mlir", "w") do io
    write(io, string(code))
end

# ╔═╡ a18a6143-a2c9-417f-8e6f-7f839815e99c
string(code)

# ╔═╡ e8920b2f-36b7-4c4f-adaa-e3abd556f3e4
@code_xla Lux.apply(raw_model,dumy_input,ps,st)

# ╔═╡ Cell order:
# ╠═6020c872-058e-11f0-325a-53e3477ce60c
# ╠═9e80b73c-92ac-4583-b6ae-bfc94f3ea429
# ╠═53dd5feb-0836-4aff-9c8e-67922cfc548e
# ╠═74e95b14-7440-4301-b3ec-4578eaab5cf2
# ╠═e6a28bfa-353b-4985-882b-97357f1ef46d
# ╠═2b427a00-c730-4632-bb12-b8c4049d2583
# ╠═15ecb8ac-f4a8-4331-9cde-829f60be5f26
# ╠═af81d80c-d75b-42ef-84a8-b59ae6d99cff
# ╠═c081367e-219f-4092-82c2-c78c99622986
# ╠═45a8d4da-9701-4568-a2d6-64eff31baee3
# ╠═5a1d21a3-835f-4f2c-8326-a12c174f7392
# ╠═227239ed-ef7d-417c-925d-faab55c09c95
# ╠═d9823229-89d4-4310-baeb-74894de7efac
# ╠═27d4d940-6842-400d-b496-102be93f79c7
# ╠═7650e69e-f022-45f3-9ac1-92602cd0308a
# ╠═8ec24e83-2a87-4040-9430-a79b371331c7
# ╠═b369794d-6b43-4c86-bbf7-42f7d35d6919
# ╠═b3a4a3a2-f7eb-4d6c-8718-35c2d36658df
# ╠═a18a6143-a2c9-417f-8e6f-7f839815e99c
# ╠═e8920b2f-36b7-4c4f-adaa-e3abd556f3e4
