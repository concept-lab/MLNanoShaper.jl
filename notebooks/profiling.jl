### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 28210014-39eb-11ef-24eb-110acb81da08
using Pkg;Pkg.activate(".")

# ╔═╡ e9f0f433-0fe9-4096-b484-b432ec54afc8
using MLNanoShaper, MLNanoShaperRunner, FileIO, StructArrays, Static, Serialization,
      GeometryBasics, LuxCUDA, Lux, Profile, ProfileSVG, ChainRulesCore, Folds,
      BenchmarkTools, Zygote, Distances, LinearAlgebra, LoopVectorization, Folds,
      StaticTools, PProf, CUDA, Adapt, NearestNeighbors, MarchingCubes, FileIO, Transducers,Accessors, Revise

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
	mins = atoms.start .-2
	maxes = mins .+ size(atoms.grid) .* atoms.radius .+2
    ranges = range.(mins, maxes; step)
    grid = Point3f.(reshape(ranges[1], :, 1,1), reshape(ranges[2], 1, :,1), reshape(ranges[3], 1,1,:))
	volume = Folds.map(grid) do x
        model((Batch([x]), atoms)) |> only
    end
	volume
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

# ╔═╡ 47641b85-0596-4ce4-992b-6811ff89574b
nthreads()

# ╔═╡ 5765cbc5-ec12-406e-b43f-9291c99b9d1d
prot_num = 1

# ╔═╡ 2cce8a1a-97fe-45ae-bca7-584b843739d6
surface = load("$(homedir())/datasets/pqr/$prot_num/triangulatedSurf.off")

# ╔═╡ d9898910-1b29-400c-bcea-457017723c70
atoms = RegularGrid(
    getfield.(
        read("$(homedir())/datasets/pqr/$prot_num/structure.pqr", PQR{Float32}), :pos) |>
    StructVector,3f0)

# ╔═╡ 08abee3c-49ee-42a1-adae-7b5a7d09a8f5
#@benchmark MLNanoShaperRunner._inrange(atoms,Point3f(10,22,0)) 

# ╔═╡ f2343acb-23c2-4155-b393-c0bcea4d9760
#@benchmark model((MLNanoShaperRunner.Batch([Point3f(10,22,0)]),atoms))

# ╔═╡ 187ca178-6b4c-406b-b312-81e12026b720
#@benchmark model((MLNanoShaperRunner.Batch([Point3f(10,22,0),Point3f(10,22,0)]),atoms))

# ╔═╡ f49f8d87-540b-4767-bbb2-794cab2da54a
#@benchmark model((MLNanoShaperRunner.Batch([Point3f(10,22,0) for _ in 1:1000]),atoms))

# ╔═╡ f723350c-c5c2-428b-9617-4db6722d87f7
#@benchmark model((MLNanoShaperRunner.Batch([Point3f(10,22,0) for _ in 1:1000_000]),atoms))

# ╔═╡ cb7a84e8-d5ae-4f43-a1ca-bd047f39dd2b
#@benchmark model_gpu((MLNanoShaperRunner.Batch([Point3f(10,22,0) for _ in 1:1000]),atoms))

# ╔═╡ fac62889-166c-43dc-8bb9-210f217ebcb0
#@benchmark model.model.layers[1].fun((MLNanoShaperRunner.Batch([Point3f(10,22,0) for _ in 1:1000]),atoms))

# ╔═╡ c97eb631-f5fa-4f13-a7bf-2bc874c527d2
CUDA.@profile model_gpu((MLNanoShaperRunner.Batch([Point3f(10,22,0) for _ in 1:1000]),atoms))

# ╔═╡ 3817b4a9-18ea-490f-9bf0-2b9593478bdd
14.3 - 11.6

# ╔═╡ c89ef485-4986-4c9e-adc4-1e8019a872cb
11.6 - 8.3

# ╔═╡ d634d86e-3db5-4d25-8177-0fcb7ff18643
77/20

# ╔═╡ bb374966-08bf-420b-a486-eba42ad359ce
@benchmark model.model.layers[1].fun((MLNanoShaperRunner.Batch([Point3f(10,22,0) for _ in 1:10]),atoms))

# ╔═╡ 376569b8-1225-4b44-9eae-62bdba87eed1
#@benchmark MLNanoShaperRunner.select_neighboord(Point3f(10,22,0),atoms)

# ╔═╡ 387b3778-a710-4bba-9942-b6119d1561da
length(atoms.grid) * 2e-6 * 3*3*3 / 12

# ╔═╡ b4cb025a-e473-4feb-aace-a533503c3672
model_weights = deserialize("$(homedir())/datasets/models/tiny_angular_dense_s_jobs_14_6_3_c_2025-03-19_epoch_400_9592899277305186470")

# ╔═╡ a820838f-7105-4770-8a26-b4cb4af3bec1
model_gpu = Lux.StatefulLuxLayer{true}(model_weights.model(on_gpu=true),model_weights.parameters |> gpu_device(),model_weights.states)

# ╔═╡ e6f8e419-2fb4-4c8a-afd3-05e500553cfc
model_fixed_size = Lux.StatefulLuxLayer{true}(model_weights.model(max_nb_atoms = 10),model_weights.parameters,model_weights.states)

# ╔═╡ 5e6d15a4-c551-40bd-97ed-57c787734217
gpu_device()

# ╔═╡ 0adf29e7-a6e4-48ae-bfe0-e5340d1d1a70
model = MLNanoShaperRunner.production_instantiate(model_weights)

# ╔═╡ c114a7fc-12bd-4a4c-99c9-c665145eaf92
model((Batch([Point3f(10,22.65,0),Point3f(0,0,0),Point3f(0,0,0)]),atoms))

# ╔═╡ a29d5a40-6c4b-4a48-98c3-176f1a3a0591
begin
	vol = evaluate_model(model,atoms;step)
	mins = atoms.start .-2 |> collect
	maxes = mins .+ size(atoms.grid) .* Float32(atoms.radius) .+2
	x,y,z = range.(mins,maxes;step) .|> collect .|> Vector{Float32}
	mc = MC(vol;x,y,z)
	march(mc,.5)
	msh = MarchingCubes.makemesh(GeometryBasics, mc)
	msh
	Mk.mesh(msh; color = :red)
end

# ╔═╡ f7bd50f3-607f-419f-bd64-c7c6e938bf14
vol

# ╔═╡ 87ebe306-cda2-49f3-980d-e303fd1244b7
# ╠═╡ skip_as_script = true
#=╠═╡
write_off("predicted.off",msh)
  ╠═╡ =#

# ╔═╡ c4b601eb-8f31-4d07-84a4-46c58ee81e6b
abs.(MLNanoShaperRunner.evaluate_field(model,atoms) .- evaluate_model(model,atoms)) |> maximum

# ╔═╡ 0d81f6cb-c6d4-4748-b23c-46ab1b0f9a8e
cutoff_radius = 3.0f0

# ╔═╡ 2ff3238a-2205-405a-9075-553f27db84d6
default_value = -8.0f0

# ╔═╡ cfea79c8-b72f-462c-ac17-6ab259677430
@benchmark cu([1,2])

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
    Profile.init(n = 10^6, delay = .05*10^-6)
    @profile  [model((MLNanoShaperRunner.Batch([Point3f(10,22,0) for _ in 1:1000]),atoms)) for _ in 1:100] 
	pprof()
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═e4fc0299-2b72-4b8f-940d-9f55a76f83ca
# ╠═28210014-39eb-11ef-24eb-110acb81da08
# ╠═1c0a8115-da6e-4b09-a9ac-17672c8b73d2
# ╠═e9f0f433-0fe9-4096-b484-b432ec54afc8
# ╠═e4a81477-34da-4891-9d0e-34a30ada4ac3
# ╠═9e4b837a-d031-41c9-b3f8-f079b0a6d16b
# ╠═c114a7fc-12bd-4a4c-99c9-c665145eaf92
# ╠═c6e28039-0412-4327-a4a0-b4df80b5ef78
# ╠═e6b81054-46b2-4854-be41-ca522bead47a
# ╠═a29d5a40-6c4b-4a48-98c3-176f1a3a0591
# ╠═f7bd50f3-607f-419f-bd64-c7c6e938bf14
# ╠═87ebe306-cda2-49f3-980d-e303fd1244b7
# ╠═47641b85-0596-4ce4-992b-6811ff89574b
# ╠═5765cbc5-ec12-406e-b43f-9291c99b9d1d
# ╠═2cce8a1a-97fe-45ae-bca7-584b843739d6
# ╠═d9898910-1b29-400c-bcea-457017723c70
# ╠═c4b601eb-8f31-4d07-84a4-46c58ee81e6b
# ╠═08abee3c-49ee-42a1-adae-7b5a7d09a8f5
# ╠═f2343acb-23c2-4155-b393-c0bcea4d9760
# ╠═187ca178-6b4c-406b-b312-81e12026b720
# ╠═f49f8d87-540b-4767-bbb2-794cab2da54a
# ╠═f723350c-c5c2-428b-9617-4db6722d87f7
# ╠═cb7a84e8-d5ae-4f43-a1ca-bd047f39dd2b
# ╠═fac62889-166c-43dc-8bb9-210f217ebcb0
# ╠═c97eb631-f5fa-4f13-a7bf-2bc874c527d2
# ╠═3817b4a9-18ea-490f-9bf0-2b9593478bdd
# ╠═c89ef485-4986-4c9e-adc4-1e8019a872cb
# ╠═d634d86e-3db5-4d25-8177-0fcb7ff18643
# ╠═bb374966-08bf-420b-a486-eba42ad359ce
# ╠═376569b8-1225-4b44-9eae-62bdba87eed1
# ╠═387b3778-a710-4bba-9942-b6119d1561da
# ╠═b4cb025a-e473-4feb-aace-a533503c3672
# ╠═a820838f-7105-4770-8a26-b4cb4af3bec1
# ╠═e6f8e419-2fb4-4c8a-afd3-05e500553cfc
# ╠═5e6d15a4-c551-40bd-97ed-57c787734217
# ╠═0adf29e7-a6e4-48ae-bfe0-e5340d1d1a70
# ╠═0d81f6cb-c6d4-4748-b23c-46ab1b0f9a8e
# ╠═2ff3238a-2205-405a-9075-553f27db84d6
# ╠═cfea79c8-b72f-462c-ac17-6ab259677430
# ╠═f44f276c-7de9-49dc-b584-5a64a04006a0
# ╠═70663eda-5bd3-4f08-8792-5f848edccaff
# ╠═1b16179f-8da8-4c34-9455-5554a3151f40
# ╠═988caa2b-a9cf-4226-beb2-52efa750beca
# ╠═565892fa-df36-4fad-8872-a9e2f7de684f
# ╠═34d53b3e-0f9e-4088-aa7d-b1acf8516e4b
# ╠═5273eaf5-2762-41ba-b634-17e170adc65e
