### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ e19e8862-2019-11f0-3dcc-fbb75f45112a
using Pkg; Pkg.activate(".")

# ╔═╡ f8b89cc9-2e7a-46d0-ae96-be746ff61543
using MLNanoShaper, MLNanoShaperRunner, Zygote,GeometryBasics, Serialization, StructArrays, BenchmarkTools, PProf, Profile, CUDA

# ╔═╡ 6882064c-0fcd-4257-a423-289a818aa4b6
prot_num = 1

# ╔═╡ 89d99e2d-11e5-452b-b2b2-e0068fa64ed9
atoms = RegularGrid(
    getfield.(
        read("$(homedir())/datasets/pqr/$prot_num/structure.pqr", PQR{Float32}), :pos) |>
    StructVector,3f0)

# ╔═╡ af5d4953-53c3-4abc-aed7-87f18dde3447
atoms.grid .|> length |> sum

# ╔═╡ bd6186c6-e3aa-4366-be91-41984a1716ad
model_weights = deserialize("$(homedir())/datasets/models/tiny_angular_dense_s_jobs_14_6_3_c_2025-03-19_epoch_400_9592899277305186470")

# ╔═╡ 0292f0af-6e27-4a3a-932c-31e0f708d4d8
model = MLNanoShaperRunner.production_instantiate(model_weights)

# ╔═╡ ade445af-19fd-4637-9c9f-f4aa0c2b577f
model_raw = MLNanoShaperRunner.drop_preprocessing(model.model)

# ╔═╡ afa5fca7-8b61-4f3c-bafb-4bdd0d76a77a
model_raw_gpu = MLNanoShaperRunner.drop_preprocessing(MLNanoShaperRunner.production_instantiate(model_weights,on_gpu=true).model)

# ╔═╡ 1757532d-8997-469b-86ff-d8e838bb3ef5
ps = model.ps

# ╔═╡ d63483b2-23dc-40fe-8b0f-505e6d7bba61
st = model.st

# ╔═╡ 64a3e6fe-6270-44dd-a102-4ded1d00c891
model_preprocessing = MLNanoShaperRunner.get_preprocessing(model.model)

# ╔═╡ 1bc1b961-0980-46e1-b614-884e990713da
@benchmark MLNanoShaperRunner._inrange(Matrix{Sphere{Float32}},atoms,Batch([Point3f(10,22,0) for _ in 1:1000]))

# ╔═╡ 273e0b3c-1ba6-4fe1-998a-76a61172f1a5
atoms_selection = MLNanoShaperRunner._inrange(Matrix{Sphere{Float32}},atoms,Batch([Point3f(10,22,0) for _ in 1:10000])) .|> StructVector |> Batch

# ╔═╡ 968f3b06-f0f9-451e-9940-b2fb923312e7
points = Batch([Point3f(10,22,0) for _ in 1:10000])

# ╔═╡ a62097e7-4d89-4b25-9320-bbfba1b7c077
_points = cu(points)

# ╔═╡ 918ef05c-2113-404b-ae3e-097c71cdc431
_atoms_selection = ConcatenatedBatch(atoms_selection) 

# ╔═╡ 3ace9b22-bc2c-4b4e-a03e-221f255aae1f
@benchmark CUDA.@sync MLNanoShaperRunner.preprocessing(_points,_atoms_selection;cutoff_radius = 3f0)

# ╔═╡ 180f3a31-0336-4fa4-b43d-8f5284ee2e0d
@benchmark MLNanoShaperRunner.preprocessing(points,atoms_selection;cutoff_radius = 3f0)

# ╔═╡ 130af005-d02f-4f2e-8b0b-aec1b4bea7e5
13e6/105000

# ╔═╡ b91b2ad1-1e2c-46cf-a4b5-f4313fef509a
 MLNanoShaperRunner.preprocessing(points,atoms_selection;cutoff_radius = 3f0)

# ╔═╡ 7711389e-dfa8-4499-8c1c-4d04b959d067
selection = MLNanoShaperRunner._inrange(Matrix{Sphere{Float32}},atoms,Batch([Point3f(10,22,0) for _ in 1:10000]))

# ╔═╡ 3893e9e9-97c8-490b-8e59-2dc24b6cf6c7
data = model_preprocessing((MLNanoShaperRunner.Batch([Point3f(10,22,0) for _ in 1:100000]),atoms),ps,st)|> first

# ╔═╡ 379589f8-65e4-44f1-bddb-7b16766a2dc3
Base.summarysize(data) / 1024^3

# ╔═╡ 15bca53a-4906-484a-8999-bb7cabcd5666
_ps = cu(ps)

# ╔═╡ b445f8d5-190b-4c41-8378-fd4ff8646fe3
_st = cu(st)

# ╔═╡ 2ab8ff18-ddd1-40ab-baa6-56da5baca148
_data = cu(data)

# ╔═╡ 71898589-3344-4687-b188-42ab629599c2
@benchmark model_raw(data,ps,st)

# ╔═╡ b6f562fb-2f1f-4b8f-9dda-e26a5f122e7b
@benchmark CUDA.@sync model_raw(_data,_ps,_st)

# ╔═╡ d70e9cde-9bd8-4bdc-9629-fea1539693e9
@benchmark gradient(ps) do ps model_raw(data,ps,st) |> first |> sum |> only end

# ╔═╡ 779b2cfe-f0a5-452a-90c1-172c77986ed5
@benchmark CUDA.@sync  gradient(_ps)do _ps model_raw(_data,_ps,_st) |> first |> sum |> only end

# ╔═╡ 306c89a9-a6ee-4066-b469-db5ede3feea7
dummy_input = zeros(Float32,4,size(data.field,2))

# ╔═╡ 1a66afc1-26e0-4f27-afb7-e93dda0d26f0
CUDA.@profile CUDA.@sync model_raw(_data,_ps,_st) |> first |> sum |> only 

# ╔═╡ 6e384744-f31c-4df5-9688-d416fa185bc6
CUDA.@profile CUDA.@sync MLNanoShaperRunner.preprocessing(_points,_atoms_selection;cutoff_radius = 3f0)

# ╔═╡ 51ed4739-8e7d-4e94-8a57-591c2f1cd5a2
begin
    Profile.clear()
    Profile.init(n = 10^6, delay = 10^-6)
    @profile [gradient(_ps) do ps
	model_raw(_data,_ps,_st) |> first |> sum |> only
end for _ in 1:10] 
	pprof() 
end

# ╔═╡ c7ab6873-1d36-448e-b363-5c4580bae782
CUDA.@profile CUDA.@sync gradient(_ps) do _ps
	model_raw(_data,_ps,_st) |> first |> sum |> only
end 

# ╔═╡ Cell order:
# ╠═e19e8862-2019-11f0-3dcc-fbb75f45112a
# ╠═f8b89cc9-2e7a-46d0-ae96-be746ff61543
# ╠═6882064c-0fcd-4257-a423-289a818aa4b6
# ╠═89d99e2d-11e5-452b-b2b2-e0068fa64ed9
# ╠═af5d4953-53c3-4abc-aed7-87f18dde3447
# ╠═bd6186c6-e3aa-4366-be91-41984a1716ad
# ╠═0292f0af-6e27-4a3a-932c-31e0f708d4d8
# ╠═ade445af-19fd-4637-9c9f-f4aa0c2b577f
# ╠═afa5fca7-8b61-4f3c-bafb-4bdd0d76a77a
# ╠═1757532d-8997-469b-86ff-d8e838bb3ef5
# ╠═d63483b2-23dc-40fe-8b0f-505e6d7bba61
# ╠═64a3e6fe-6270-44dd-a102-4ded1d00c891
# ╠═1bc1b961-0980-46e1-b614-884e990713da
# ╠═273e0b3c-1ba6-4fe1-998a-76a61172f1a5
# ╠═968f3b06-f0f9-451e-9940-b2fb923312e7
# ╠═a62097e7-4d89-4b25-9320-bbfba1b7c077
# ╠═918ef05c-2113-404b-ae3e-097c71cdc431
# ╠═3ace9b22-bc2c-4b4e-a03e-221f255aae1f
# ╠═180f3a31-0336-4fa4-b43d-8f5284ee2e0d
# ╠═130af005-d02f-4f2e-8b0b-aec1b4bea7e5
# ╠═b91b2ad1-1e2c-46cf-a4b5-f4313fef509a
# ╠═7711389e-dfa8-4499-8c1c-4d04b959d067
# ╠═3893e9e9-97c8-490b-8e59-2dc24b6cf6c7
# ╠═379589f8-65e4-44f1-bddb-7b16766a2dc3
# ╠═15bca53a-4906-484a-8999-bb7cabcd5666
# ╠═b445f8d5-190b-4c41-8378-fd4ff8646fe3
# ╠═2ab8ff18-ddd1-40ab-baa6-56da5baca148
# ╠═71898589-3344-4687-b188-42ab629599c2
# ╠═b6f562fb-2f1f-4b8f-9dda-e26a5f122e7b
# ╠═d70e9cde-9bd8-4bdc-9629-fea1539693e9
# ╠═779b2cfe-f0a5-452a-90c1-172c77986ed5
# ╠═306c89a9-a6ee-4066-b469-db5ede3feea7
# ╠═1a66afc1-26e0-4f27-afb7-e93dda0d26f0
# ╠═6e384744-f31c-4df5-9688-d416fa185bc6
# ╠═51ed4739-8e7d-4e94-8a57-591c2f1cd5a2
# ╠═c7ab6873-1d36-448e-b363-5c4580bae782
