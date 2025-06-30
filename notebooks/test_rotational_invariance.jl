### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 1a83aa26-5103-11f0-1b42-75a2e3979113
using Pkg;Pkg.activate(".")

# ╔═╡ 0e5018db-c6dc-454a-af2f-68e90cefd5fd
using MLNanoShaper, MLNanoShaperRunner, FileIO, StructArrays, Static, Serialization,
      GeometryBasics, LuxCUDA, Lux, Profile, ProfileSVG, ChainRulesCore, Folds,
      BenchmarkTools, Zygote, Distances, LinearAlgebra, LoopVectorization, Folds,
      StaticTools, PProf, CUDA, Adapt, NearestNeighbors, MarchingCubes, FileIO, Transducers,Accessors, Revise, Rotations

# ╔═╡ 5e2e0943-e132-4a02-a1d5-80d9c135a9ec
model_name = "tiny_soft_max_angular_dense_jobs_40_3_2025-06-04_epoch_4370_75456560920357034"

# ╔═╡ 7afc7b48-8518-4c98-833f-4bfd41d3acdb
model_weights = deserialize("$(homedir())/datasets/models/$model_name")

# ╔═╡ e99a9a89-e9ec-4ebb-a009-08f7417db580
x = [Point3f(0,1,0),Point3f(1,0,0),Point3f(10,13,-22)]

# ╔═╡ 1f9d542b-e969-4128-95ff-ddca0df67a3a
r = RotationVec{Float32}(3,π/2,7)

# ╔═╡ f9590acb-e917-4f22-b7a3-8d80ec1772d3
k = []

# ╔═╡ 5d37fa12-4006-42a2-8637-0d71ce278960
g = x ->(push!(k,x);false)

# ╔═╡ 904ff768-5c28-4171-8054-b2a685fe2180
g(1)

# ╔═╡ d20939ce-4a47-4c40-b7e1-24e825557c19
k

# ╔═╡ e3ea2151-9961-4f87-ab7e-d7e48354dffa
MLNanoShaperRunner.Δ3

# ╔═╡ 8e0b4c1f-1329-4339-80bc-edce706f49e7
model = MLNanoShaperRunner.production_instantiate(model_weights,on_gpu=false)

# ╔═╡ dce7a585-fd94-45fc-acc1-38bbcfd3c9e4
f = MLNanoShaperRunner.get_preprocessing(model.model).fun

# ╔═╡ 9127ebf8-bb02-4143-8fd9-ea9743003ccb
prot_num = 2

# ╔═╡ 2cf10ec9-616b-43b9-aee7-bec1470eeccd
list_atoms = getfield.(
        read("$(homedir())/datasets/pqr/$prot_num/structure.pqr", PQR{Float32}), :pos)

# ╔═╡ f2d65f48-4743-40f3-ad63-20804e9bf06c
atoms_rotated = RegularGrid(map(list_atoms) do a
	Sphere(r*a.center,a.r)
end|>
    StructVector,3f0)

# ╔═╡ 2ef4ea85-87d6-4a82-9d5e-d9330adb5c95
ŷ = MLNanoShaperRunner._inrange.(StructVector{Sphere{Float32}},Ref(atoms_rotated),Ref(r) .*x[3:3])

# ╔═╡ 50e27729-6b1e-49fa-911d-185f945dae96
Ref(r) .\ (only(ŷ)).center 

# ╔═╡ 51696178-456f-446a-b267-964ea581d164
atoms = RegularGrid(list_atoms |>StructVector,3f0)

# ╔═╡ 94d3ac4d-1cfa-48a7-aacc-5da7668309a3
atoms.radius^2

# ╔═╡ a573bdf2-ac36-460d-8f81-f170d3b6b314
y = MLNanoShaperRunner._inrange.(StructVector{Sphere{Float32}},Ref(atoms),x[3:3])

# ╔═╡ 3d1e065e-a005-4a53-9464-1ec72a74a68d
(only(y)).center

# ╔═╡ 01d63e6d-b4e9-4058-a0aa-9184623cea3f
(only(y)).r

# ╔═╡ c8dc0c35-71f0-46bb-b130-30b614030f52
y .- ŷ

# ╔═╡ Cell order:
# ╠═1a83aa26-5103-11f0-1b42-75a2e3979113
# ╠═0e5018db-c6dc-454a-af2f-68e90cefd5fd
# ╠═5e2e0943-e132-4a02-a1d5-80d9c135a9ec
# ╠═7afc7b48-8518-4c98-833f-4bfd41d3acdb
# ╠═e99a9a89-e9ec-4ebb-a009-08f7417db580
# ╠═2cf10ec9-616b-43b9-aee7-bec1470eeccd
# ╠═f2d65f48-4743-40f3-ad63-20804e9bf06c
# ╠═1f9d542b-e969-4128-95ff-ddca0df67a3a
# ╠═94d3ac4d-1cfa-48a7-aacc-5da7668309a3
# ╠═3d1e065e-a005-4a53-9464-1ec72a74a68d
# ╠═01d63e6d-b4e9-4058-a0aa-9184623cea3f
# ╠═50e27729-6b1e-49fa-911d-185f945dae96
# ╠═5d37fa12-4006-42a2-8637-0d71ce278960
# ╠═f9590acb-e917-4f22-b7a3-8d80ec1772d3
# ╠═904ff768-5c28-4171-8054-b2a685fe2180
# ╠═d20939ce-4a47-4c40-b7e1-24e825557c19
# ╠═a573bdf2-ac36-460d-8f81-f170d3b6b314
# ╠═e3ea2151-9961-4f87-ab7e-d7e48354dffa
# ╠═2ef4ea85-87d6-4a82-9d5e-d9330adb5c95
# ╠═dce7a585-fd94-45fc-acc1-38bbcfd3c9e4
# ╠═c8dc0c35-71f0-46bb-b130-30b614030f52
# ╠═8e0b4c1f-1329-4339-80bc-edce706f49e7
# ╠═9127ebf8-bb02-4143-8fd9-ea9743003ccb
# ╠═51696178-456f-446a-b267-964ea581d164
