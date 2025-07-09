### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 7b259d1e-1132-11f0-30c6-c9559109859f
using Pkg;Pkg.activate(".")

# ╔═╡ 47e986dd-7fee-4e55-85ff-c63a592fc6ca
using Revise

# ╔═╡ 829ae9d2-105a-4a98-ad56-e0016b4f04d9
using MLNanoShaper, MLNanoShaperRunner, FileIO, StructArrays, Static, Serialization,
      GeometryBasics, LuxCUDA, Lux, Profile, ProfileSVG, ChainRulesCore, Folds,
      BenchmarkTools, Zygote, Distances, LinearAlgebra, LoopVectorization, Folds,
      StaticTools, PProf, CUDA, Adapt, NearestNeighbors, MarchingCubes, FileIO, Transducers,Accessors 

# ╔═╡ 671c0869-ecf4-48be-a22c-7e373bebc294
using Base.Threads

# ╔═╡ 05145ea8-59b9-41fb-ad7e-fce08fa0c36c
import CairoMakie as Mk

# ╔═╡ 880fbf4e-0951-400e-a997-d4f6ecf72ad1
Threads.nthreads()

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
step =.5f0

# ╔═╡ ddba2880-fff8-4df2-87e1-67b8c87bcd72
prot_num = 1

# ╔═╡ 498adb72-6edf-4b36-b9a5-474f6660a030
model_name = "tiny_soft_max_angular_dense_testhardsigma5_20000_17558598383046663794"

# ╔═╡ 43cc30fc-c266-4cf1-aff0-c5c505cf4924
model_weights = deserialize("$(homedir())/datasets/models/$model_name")

# ╔═╡ 2b75694a-d5ae-45f3-93af-61c4167314d9
model = MLNanoShaperRunner.production_instantiate(model_weights,on_gpu=true)

# ╔═╡ 4f55720b-73d4-4b29-80d3-43c74f0f48bc
vec_atoms = getfield.(
        read("$(homedir())/datasets/pqr/$prot_num/structure.pqr", PQR{Float32}), :pos) |>
    StructVector

# ╔═╡ 634427ef-6126-4fdd-a8b2-3d0bfee0d0b6
atoms = RegularGrid(vec_atoms,3f0)

# ╔═╡ ce214ac0-f811-4382-beac-b0ba82b0e206
read("$(homedir())/datasets/pqr/$prot_num/structure.pqr", PQR{Float32}) |> length

# ╔═╡ 8d39769c-9e87-4f5b-aa50-34abe8c78cf5
vol = MLNanoShaperRunner.evaluate_field_fast(model,vec_atoms;step) 

# ╔═╡ ab1823fe-a0b3-4f34-a7c3-aa7d3e507efe
Threads.nthreads()

# ╔═╡ 3f5f7acc-5f9a-47a1-9ab9-084abee612f2
#@benchmark MLNanoShaperRunner.evaluate_field_fast(model,atoms;step) 

# ╔═╡ 206cb7a0-bc63-4f3d-b0f6-cf7255cea696
vol1 = MLNanoShaperRunner.evaluate_field(model,atoms;step)

# ╔═╡ 439948ac-0a40-4ce6-ad07-b139c73e053d
maximum(vol .- vol1)

# ╔═╡ cdd55fc4-fccd-4dd7-b0dc-3b7b65804334
mean((vol .- mean(vol)) .* (vol1 .- mean(vol1)))

# ╔═╡ 1c843626-0390-4bea-819f-114506062d3b
sqrt(mean((vol .- vol1).^2))

# ╔═╡ 81d5a6d9-0030-45d2-83b7-f1b043fed0f9
length(vol) / 1000

# ╔═╡ 315999e7-1a22-4cc1-a0da-5094a5843015
function get_mesh(vol,atoms)
	mins = atoms.start .-2 |> collect
	maxes = mins .+ size(atoms.grid) .* Float32(atoms.radius) .+2
	x,y,z = range.(mins,maxes;step) .|> collect .|> Vector{Float32}
	mc = MC(vol;x,y,z)
	march(mc,.5)
	MarchingCubes.makemesh(GeometryBasics, mc)
end

# ╔═╡ f8a03541-2938-42c5-8c18-30cfdae902e9
msh = get_mesh(vol,atoms)

# ╔═╡ 9cff681f-a05e-4125-b070-35a831a26321
msh1 = get_mesh(vol1,atoms) 

# ╔═╡ c7cf7dd4-148c-40c4-ada4-fa6935348c7f
write_off("$model_name-predicted.off",msh1)

# ╔═╡ 1595185f-054b-42bb-a651-1eb925bf31fe
Mk.mesh(msh; color = :red)

# ╔═╡ c0b55cc0-02bb-44f6-aaa5-bba81f0019f8
Mk.mesh(msh1; color = :red)

# ╔═╡ 1ffca824-d203-47e1-b140-e59cbd7ef655
write_off("$model_name-predicted_fast.off",msh)

# ╔═╡ Cell order:
# ╠═7b259d1e-1132-11f0-30c6-c9559109859f
# ╠═05145ea8-59b9-41fb-ad7e-fce08fa0c36c
# ╠═47e986dd-7fee-4e55-85ff-c63a592fc6ca
# ╠═829ae9d2-105a-4a98-ad56-e0016b4f04d9
# ╠═671c0869-ecf4-48be-a22c-7e373bebc294
# ╠═880fbf4e-0951-400e-a997-d4f6ecf72ad1
# ╠═b1188b32-f4ba-44c0-995e-070ff2505888
# ╠═0f509aaf-d162-4b90-908f-0958a2846245
# ╠═ddba2880-fff8-4df2-87e1-67b8c87bcd72
# ╠═498adb72-6edf-4b36-b9a5-474f6660a030
# ╠═43cc30fc-c266-4cf1-aff0-c5c505cf4924
# ╠═2b75694a-d5ae-45f3-93af-61c4167314d9
# ╠═4f55720b-73d4-4b29-80d3-43c74f0f48bc
# ╠═634427ef-6126-4fdd-a8b2-3d0bfee0d0b6
# ╠═ce214ac0-f811-4382-beac-b0ba82b0e206
# ╠═8d39769c-9e87-4f5b-aa50-34abe8c78cf5
# ╠═ab1823fe-a0b3-4f34-a7c3-aa7d3e507efe
# ╠═3f5f7acc-5f9a-47a1-9ab9-084abee612f2
# ╠═206cb7a0-bc63-4f3d-b0f6-cf7255cea696
# ╠═439948ac-0a40-4ce6-ad07-b139c73e053d
# ╠═cdd55fc4-fccd-4dd7-b0dc-3b7b65804334
# ╠═1c843626-0390-4bea-819f-114506062d3b
# ╠═81d5a6d9-0030-45d2-83b7-f1b043fed0f9
# ╠═315999e7-1a22-4cc1-a0da-5094a5843015
# ╠═c7cf7dd4-148c-40c4-ada4-fa6935348c7f
# ╠═f8a03541-2938-42c5-8c18-30cfdae902e9
# ╠═9cff681f-a05e-4125-b070-35a831a26321
# ╠═1595185f-054b-42bb-a651-1eb925bf31fe
# ╠═c0b55cc0-02bb-44f6-aaa5-bba81f0019f8
# ╠═1ffca824-d203-47e1-b140-e59cbd7ef655
