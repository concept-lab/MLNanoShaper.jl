### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 28210014-39eb-11ef-24eb-110acb81da08
using Pkg

# ╔═╡ de53cb27-dbef-4e3d-9d12-0f3d1b5acbc4
Pkg.activate(".")

# ╔═╡ e9f0f433-0fe9-4096-b484-b432ec54afc8
using MLNanoShaper, MLNanoShaperRunner, FileIO, StructArrays, Static, Serialization,
      GeometryBasics, LuxCUDA, Lux, Profile, ProfileSVG, ChainRulesCore, Folds,
      BenchmarkTools, Zygote, Distances, LinearAlgebra, LoopVectorization, Folds,
      StaticTools, PProf, CUDA, Adapt, NearestNeighbors, MarchingCubes, FileIO, Transducers 

# ╔═╡ e4a81477-34da-4891-9d0e-34a30ada4ac3
using Base.Threads

# ╔═╡ e4fc0299-2b72-4b8f-940d-9f55a76f83ca
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

# ╔═╡ 1c0a8115-da6e-4b09-a9ac-17672c8b73d2
import CairoMakie as Mk

# ╔═╡ 9e4b837a-d031-41c9-b3f8-f079b0a6d16b
function evaluate_model(model,atoms;step=1)
	(; mins, maxes) = atoms.tree.hyper_rec
    ranges = range.(mins, maxes; step)
    grid = Point3f.(reshape(ranges[1], :, 1,1), reshape(ranges[2], 1, :,1), reshape(ranges[3], 1,1,:))
	atoms = atoms
	Folds.map(grid) do x
        model((MLNanoShaper.Batch([x]), atoms)) |> only
    end
end

# ╔═╡ c6e28039-0412-4327-a4a0-b4df80b5ef78
step = 1

# ╔═╡ e6b81054-46b2-4854-be41-ca522bead47a
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

# ╔═╡ f7942c8f-8e0e-4e88-9514-80aac9b7cfa9
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	#vol = evaluate_model(model,atoms;step)
	#(; mins, maxes) = atoms.tree.hyper_rec
	#x,y,z = range.(mins,maxes;step) .|> collect .|> Vector{Float32}
	#mc = MC(vol;x,y,z)
	#march(mc,.5)
	#msh = MarchingCubes.makemesh(GeometryBasics, mc)
	#msh
	#Mk.mesh(msh; color = :red)
end
  ╠═╡ =#

# ╔═╡ 87ebe306-cda2-49f3-980d-e303fd1244b7
# ╠═╡ skip_as_script = true
#=╠═╡
#write_off("predicted.off",msh)
  ╠═╡ =#

# ╔═╡ 47641b85-0596-4ce4-992b-6811ff89574b
nthreads()

# ╔═╡ 5765cbc5-ec12-406e-b43f-9291c99b9d1d
prot_num = 1

# ╔═╡ 2cce8a1a-97fe-45ae-bca7-584b843739d6
surface = load("$(homedir())/datasets/pqr/$prot_num/triangulatedSurf.off")

# ╔═╡ d9898910-1b29-400c-bcea-457017723c70
atoms = MLNanoShaperRunner.AnnotedKDTree(
    getfield.(
        read("$(homedir())/datasets/pqr/$prot_num/structure.pqr", PQR{Float32}), :pos) |>
    StructVector,
    static(:center))

# ╔═╡ 9b8e675a-52ed-4176-bfd8-f11ff0650ca2
grid = MLNanoShaperRunner.RegularGrid(getfield.(read("$(homedir())/datasets/pqr/$prot_num/structure.pqr", PQR{Float32}),:pos) |> StructVector,3f0)

# ╔═╡ 08abee3c-49ee-42a1-adae-7b5a7d09a8f5
@benchmark MLNanoShaperRunner._inrange(grid,Point3f(10,22,0)) 

# ╔═╡ c4b2cbd7-3be0-40d3-9ddf-d8ef7c4f81ca
@benchmark inrange(atoms.tree,Point3f(10,22,0),3) 

# ╔═╡ 376569b8-1225-4b44-9eae-62bdba87eed1
@benchmark MLNanoShaperRunner.select_neighboord(Point3f(10,22,0),atoms;cutoff_radius=3f0)

# ╔═╡ 6196a805-73d4-48e4-97ec-a413eae5c60e
atoms.tree

# ╔═╡ 0adf29e7-a6e4-48ae-bfe0-e5340d1d1a70
model = "$(homedir())/datasets/models/tiny_angular_dense_s_jobs_11_6_3_c_2025-03-10_epoch_800_10631177997949843226" |>
        deserialize |>
        MLNanoShaperRunner.production_instantiate 

# ╔═╡ f2343acb-23c2-4155-b393-c0bcea4d9760
@benchmark model((MLNanoShaperRunner.Batch([Point3f(10,22,0)]),atoms))

# ╔═╡ fac62889-166c-43dc-8bb9-210f217ebcb0
@benchmark model.model.layers[1].fun((MLNanoShaperRunner.Batch([Point3f(10,22,0)]),atoms))

# ╔═╡ 0d81f6cb-c6d4-4748-b23c-46ab1b0f9a8e
cutoff_radius = 3.0f0

# ╔═╡ 2ff3238a-2205-405a-9075-553f27db84d6
default_value = -8.0f0

# ╔═╡ 3ed5f526-99fd-4e91-b7f0-6bc7d8c67afa
atoms.tree.hyper_rec

# ╔═╡ f44f276c-7de9-49dc-b584-5a64a04006a0
51 * 56 * 48 / 4000 * 0.03

# ╔═╡ 70663eda-5bd3-4f08-8792-5f848edccaff
40 * 89 / 1000

# ╔═╡ 1b16179f-8da8-4c34-9455-5554a3151f40
89 * 4

# ╔═╡ 988caa2b-a9cf-4226-beb2-52efa750beca
300*300*300/30e-6

# ╔═╡ 565892fa-df36-4fad-8872-a9e2f7de684f
md"""
base time : 330 ms

modified kernel: 150 ms

modified preprocessing: 130 ms

map to gpu before symetrize : 89 ms

using malloc in preprocessing : 78 ms
"""

# ╔═╡ 34d53b3e-0f9e-4088-aa7d-b1acf8516e4b
330 / 89

# ╔═╡ 5273eaf5-2762-41ba-b634-17e170adc65e
# ╠═╡ skip_as_script = true
#=╠═╡
begin
    Profile.clear()
    Profile.init(n = 10^6, delay = 10^-5)
    @profile  [model((MLNanoShaperRunner.Batch([Point3f(10,22,0)]),atoms)) for _ in 1:10^4]
	pprof()
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═e4fc0299-2b72-4b8f-940d-9f55a76f83ca
# ╠═28210014-39eb-11ef-24eb-110acb81da08
# ╠═de53cb27-dbef-4e3d-9d12-0f3d1b5acbc4
# ╠═1c0a8115-da6e-4b09-a9ac-17672c8b73d2
# ╠═e9f0f433-0fe9-4096-b484-b432ec54afc8
# ╠═e4a81477-34da-4891-9d0e-34a30ada4ac3
# ╠═9e4b837a-d031-41c9-b3f8-f079b0a6d16b
# ╠═c6e28039-0412-4327-a4a0-b4df80b5ef78
# ╠═e6b81054-46b2-4854-be41-ca522bead47a
# ╠═f7942c8f-8e0e-4e88-9514-80aac9b7cfa9
# ╠═87ebe306-cda2-49f3-980d-e303fd1244b7
# ╠═47641b85-0596-4ce4-992b-6811ff89574b
# ╠═5765cbc5-ec12-406e-b43f-9291c99b9d1d
# ╠═2cce8a1a-97fe-45ae-bca7-584b843739d6
# ╠═d9898910-1b29-400c-bcea-457017723c70
# ╠═9b8e675a-52ed-4176-bfd8-f11ff0650ca2
# ╠═08abee3c-49ee-42a1-adae-7b5a7d09a8f5
# ╠═c4b2cbd7-3be0-40d3-9ddf-d8ef7c4f81ca
# ╠═f2343acb-23c2-4155-b393-c0bcea4d9760
# ╠═fac62889-166c-43dc-8bb9-210f217ebcb0
# ╠═376569b8-1225-4b44-9eae-62bdba87eed1
# ╠═6196a805-73d4-48e4-97ec-a413eae5c60e
# ╠═0adf29e7-a6e4-48ae-bfe0-e5340d1d1a70
# ╠═0d81f6cb-c6d4-4748-b23c-46ab1b0f9a8e
# ╠═2ff3238a-2205-405a-9075-553f27db84d6
# ╠═3ed5f526-99fd-4e91-b7f0-6bc7d8c67afa
# ╠═f44f276c-7de9-49dc-b584-5a64a04006a0
# ╠═70663eda-5bd3-4f08-8792-5f848edccaff
# ╠═1b16179f-8da8-4c34-9455-5554a3151f40
# ╠═988caa2b-a9cf-4226-beb2-52efa750beca
# ╠═565892fa-df36-4fad-8872-a9e2f7de684f
# ╠═34d53b3e-0f9e-4088-aa7d-b1acf8516e4b
# ╠═5273eaf5-2762-41ba-b634-17e170adc65e
