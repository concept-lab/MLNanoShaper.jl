### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 7b259d1e-1132-11f0-30c6-c9559109859f
using Pkg;Pkg.activate(".")

# ╔═╡ 829ae9d2-105a-4a98-ad56-e0016b4f04d9
using MLNanoShaper, MLNanoShaperRunner, FileIO, StructArrays, Static, Serialization,
      GeometryBasics, LuxCUDA, Lux, Profile, ProfileSVG, ChainRulesCore, Folds,
      BenchmarkTools, Zygote, Distances, LinearAlgebra, LoopVectorization, Folds,
      StaticTools, PProf, CUDA, Adapt, NearestNeighbors, MarchingCubes, FileIO, Transducers,Accessors, Revise

# ╔═╡ 671c0869-ecf4-48be-a22c-7e373bebc294
using Base.Threads

# ╔═╡ 05145ea8-59b9-41fb-ad7e-fce08fa0c36c
import CairoMakie as Mk

# ╔═╡ b1188b32-f4ba-44c0-995e-070ff2505888
function write_off(filename::String, mesh::GeometryBasics.Mesh)
    open(filename, "w") do io
        # Write OFF header
        println(io, "OFF")
        # Write number of vertices, faces, and edges (0 for edges as it's typically not used)
        vertices = mesh.position
        faces = mesh.faces
        println(io, "$(length(vertices)) $(length(faces)) 0")
        
        # Write vertex coordinates
        for vertex in vertices
            println(io, "$(vertex[1]) $(vertex[2]) $(vertex[3])")
        end
        
        # Write faces (first number is the number of vertices in the face)
        for face in faces
            # Assuming triangular faces (3 vertices per face)
            # Note: OFF format uses 0-based indexing, while Julia uses 1-based indexing
            indices = [i-1 for i in face]  # Convert to 0-based indexing
            println(io, "3 $(indices[1]) $(indices[2]) $(indices[3])")
        end
    end
end

# ╔═╡ 0f509aaf-d162-4b90-908f-0958a2846245
step =.3f0

# ╔═╡ ddba2880-fff8-4df2-87e1-67b8c87bcd72
prot_num = 1

# ╔═╡ c595e154-3530-4dfe-b886-9fc975e0eda1
surface = load("$(homedir())/datasets/pqr/$prot_num/triangulatedSurf.off")

# ╔═╡ 43cc30fc-c266-4cf1-aff0-c5c505cf4924
model_weights = deserialize("$(homedir())/datasets/models/tiny_angular_dense_s_jobs_15_6_3_c_2025-04-03_epoch_400_4474706076735193690")

# ╔═╡ 2b75694a-d5ae-45f3-93af-61c4167314d9
model = MLNanoShaperRunner.production_instantiate(model_weights)

# ╔═╡ 634427ef-6126-4fdd-a8b2-3d0bfee0d0b6
atoms = RegularGrid(
    getfield.(
        read("$(homedir())/datasets/pqr/$prot_num/structure.pqr", PQR{Float32}), :pos) |>
    StructVector,3f0)

# ╔═╡ bf954fa9-6d15-4226-bea2-a96807c130da
begin
	vol = MLNanoShaperRunner.evaluate_field(model,atoms;step)
	mins = atoms.start .-2 |> collect
	maxes = mins .+ size(atoms.grid) .* Float32(atoms.radius) .+2
	x,y,z = range.(mins,maxes;step) .|> collect .|> Vector{Float32}
	mc = MC(vol;x,y,z)
	march(mc,.5)
	msh = MarchingCubes.makemesh(GeometryBasics, mc)
	msh
	Mk.mesh(msh; color = :red)
end

# ╔═╡ c7cf7dd4-148c-40c4-ada4-fa6935348c7f
write_off("predicted.off",msh)

# ╔═╡ Cell order:
# ╠═7b259d1e-1132-11f0-30c6-c9559109859f
# ╠═05145ea8-59b9-41fb-ad7e-fce08fa0c36c
# ╠═829ae9d2-105a-4a98-ad56-e0016b4f04d9
# ╠═671c0869-ecf4-48be-a22c-7e373bebc294
# ╠═b1188b32-f4ba-44c0-995e-070ff2505888
# ╠═0f509aaf-d162-4b90-908f-0958a2846245
# ╠═ddba2880-fff8-4df2-87e1-67b8c87bcd72
# ╠═c595e154-3530-4dfe-b886-9fc975e0eda1
# ╠═43cc30fc-c266-4cf1-aff0-c5c505cf4924
# ╠═2b75694a-d5ae-45f3-93af-61c4167314d9
# ╠═634427ef-6126-4fdd-a8b2-3d0bfee0d0b6
# ╠═bf954fa9-6d15-4226-bea2-a96807c130da
# ╠═c7cf7dd4-148c-40c4-ada4-fa6935348c7f
