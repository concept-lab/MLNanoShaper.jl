### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 7d159d76-af4e-4808-91e8-dcbaecb06a0b
using Pkg;Pkg.activate(".")

# ╔═╡ 776d1f54-eec7-452f-869a-f82a51afc1aa
using Revise

# ╔═╡ ed2aea6c-27b2-4086-b9f3-4a7012f0c564
using TOML,MLUtils, Serialization, MarchingCubes, GeometryBasics,StructArrays, Folds, Statistics, NearestNeighbors, DataFrames, CSV, FileIO, BenchmarkTools, Lux

# ╔═╡ bc34a833-938a-4f1d-a64d-e63cbe30d9ed
import MLNanoShaper as MLN, MLNanoShaperRunner as MLNR, CairoMakie as Mk

# ╔═╡ 5ce2e411-ab3a-4427-98b0-e92af67c3c7b
parms = TOML.parsefile(MLN.params_file)

# ╔═╡ 0b3aff45-cd8d-4d86-8c9f-d1b65ebf744d
trp = MLN.read_from_TOML(MLN.TrainingParameters,parms)

# ╔═╡ 050bc35e-87ee-4298-b8e5-d37546184ddb
auxp = MLN.read_from_TOML(MLN.AuxiliaryParameters,parms)

# ╔═╡ 787215b4-c496-4484-9b86-16f3d3a7c432
(;test_data) = MLN.get_dataset(trp,auxp)

# ╔═╡ 7e8ad8fd-20b8-4e25-9af5-1b582d487ec9
model_name = "tiny_soft_max_angular_dense_jobs_40_3_2025-06-04_epoch_4370_75456560920357034"

# ╔═╡ 108f4124-37d2-41c5-a3ce-c0d34ceae893
model_weights = deserialize("$(homedir())/datasets/models/$model_name") 

# ╔═╡ 63b8d015-f706-41a1-9242-e8df290257d6
model = MLNR.production_instantiate(model_weights,on_gpu=true)

# ╔═╡ 4c2e0a34-e66c-40ab-a84e-2bc35081f302
a = getobs(test_data,1).atoms

# ╔═╡ 22c7d696-50a5-47a4-a938-51745e80c5be
@benchmark _atoms = MLNR.RegularGrid(StructVector(a),3f0)

# ╔═╡ cef73df7-da84-4105-9b41-33549ba00f6a
_atoms = MLNR.RegularGrid(StructVector(a),3f0)

# ╔═╡ c78f60e9-0a6c-4a75-98a0-df6094c7a70b
#@benchmark MLNR.evaluate_field_fast(model,StructVector(a);step=.5f0) 

# ╔═╡ 35e9be7a-f171-4ac0-b7e5-c09dec2cf565
#@benchmark evaluate_trivial!(similar(grid,Float32),grid,_atoms)

# ╔═╡ a1f9711d-5dfa-4b90-b59b-36d808f36d15
#@code_native evaluate_trivial!(similar(grid,Float32),grid,_atoms)

# ╔═╡ 5f29d001-2fd6-4100-8214-7473b42a0221
function f(grid)
	for I in eachindex(IndexCartesian(), grid)
		pos =  grid[I]
	end
end

# ╔═╡ 9f7e09b6-55fa-44ae-a5e9-9a5dd395f6ce
function evaluate_trivial!(volume::AbstractArray{Float32,3},coordinates::AbstractArray{Point3f,3},atoms::MLNR.RegularGrid)::Tuple{AbstractVector{CartesianIndex{3}},AbstractVector{Point3f}}
	cutoff_radius = atoms.radius
	cutoff_radius² = cutoff_radius^2
	n = length(volume)
	k = Threads.nthreads()
	vec_unknow_indices = Matrix{CartesianIndex{3}}(undef,k,n ÷ k)
	vec_unknow_pos = Matrix{Point3f}(undef,k,n ÷ k) 
	id_last = zeros(Int,k)
	has_atoms_nearby = falses(k)
    Threads.@threads for I in eachindex(IndexCartesian(), coordinates)
		k = Threads.threadid()
		pos = coordinates[I]
		has_atoms_nearby[k] = false
		volume[I] = 0f0
		MLNR._iter_grid(atoms,pos,MLNR.Δ3) do s::Sphere{Float32}
			d² = (s.center.- pos ) .^2 |> sum
			_k = Threads.threadid()
			if d² < s.r^2
				volume[I] = 1f0
				has_atoms_nearby[_k] = false
				return true
			elseif d² < cutoff_radius² && volume[I] == 0f0
				has_atoms_nearby[_k] = true
			end
			return false
		end
		if has_atoms_nearby[k]
			# @assert volume[I] == 0f0 "got $(volume[I])"
			id_last[k] += 1
			vec_unknow_indices[k,id_last[k]]= I
			vec_unknow_pos[k,id_last[k]]  = pos
		end
	end
	unknown_indices = CartesianIndex{3}[]
	unknown_pos = Point3f[]
	for k in 1:Threads.nthreads()
		append!(unknown_indices,view(vec_unknow_indices,k,1:id_last[k]))
		append!(unknown_pos,view(vec_unknow_pos,k,1:id_last[k]))
	end
	unknown_indices,unknown_pos
end


# ╔═╡ b78e8829-dbbd-4afc-97d6-a93f40816f43
function get_grid(atoms)
	mins = atoms.start .- 2
	maxes = atoms.start .+ size(atoms.grid) .* atoms.radius .+ 2
    ranges = range.(mins, maxes; step=.5f0)
    grid = Point3f.(reshape(ranges[1], :, 1,1), reshape(ranges[2], 1, :,1), reshape(ranges[3], 1,1,:))
end

# ╔═╡ 92b23360-c00c-42c3-b515-1fced0d6d54b
grid = get_grid(_atoms)

# ╔═╡ 6c06719a-7176-4e60-aceb-beb766007db2
100^3 / 1000^2 / 10

# ╔═╡ 6527b895-6374-41eb-b4c1-ccdf96cec140
cdtempdir(f,args...;kargs...) =  mktempdir(args...;kargs...) do dir
	cd(f,dir)
end

# ╔═╡ 31e371f6-1b02-42f7-b776-854e0aef651b
function write_pqr(io::IO,atoms::AbstractVector{Sphere{Float32}})
	for (;center,r) in atoms
		x,y,z = center
		println(io,x," ",y," ",z," ",r)
	end
end

# ╔═╡ c3d1e65e-5687-11f0-1275-d98935101b84
function run_nanoshaper(atoms::AbstractVector{Sphere{Float32}})
    cdtempdir() do
        open("structure.xyzr","w") do io
			write_pqr(io,atoms)
		end
			
		conf_path = joinpath(@__DIR__, "conf.prm")
        symlink(conf_path, "conf.prm")
        command = `$(homedir())/workspace/nanoshaper/pkg_nanoshaper_0.7.8/NanoShaper conf.prm`
		withenv(
			"LD_LIBRARY_PATH" => "$(homedir())/workspace/nanoshaper/pkg_nanoshaper_0.7.8"
		) do 
        	run(pipeline(command,devnull);wait = true)
		end
		load("predicted.off")::Mesh
	end
end

# ╔═╡ 460d1857-9cf4-4cc6-a6ed-ccbc8301c609
getobs(test_data,1).atoms |> run_nanoshaper 

# ╔═╡ c75844b7-8a6b-4482-b923-14aa97bac9fa
run_nanoshaper(a)

# ╔═╡ 330f6a43-394f-4c08-8b9f-5506cb9fd4c1
function run_MLNanoShaper(atoms::AbstractVector{Sphere{Float32}})
	atoms = MLNR.RegularGrid(StructVector(atoms),3f0)
	vol = MLNR.evaluate_field_fast(model,atoms;step=.5f0)
	return vol
	mins = atoms.start .-2 |> collect
	maxes = mins .+ size(atoms.grid) .* Float32(atoms.radius) .+2
	x,y,z = range.(mins,maxes;step=.5f0) .|> collect .|> Vector{Float32}
	mc = MC(vol;x,y,z)
	march(mc,.5)
	MarchingCubes.makemesh(GeometryBasics, mc)
end

# ╔═╡ Cell order:
# ╠═7d159d76-af4e-4808-91e8-dcbaecb06a0b
# ╠═776d1f54-eec7-452f-869a-f82a51afc1aa
# ╠═bc34a833-938a-4f1d-a64d-e63cbe30d9ed
# ╠═5ce2e411-ab3a-4427-98b0-e92af67c3c7b
# ╠═ed2aea6c-27b2-4086-b9f3-4a7012f0c564
# ╠═0b3aff45-cd8d-4d86-8c9f-d1b65ebf744d
# ╠═050bc35e-87ee-4298-b8e5-d37546184ddb
# ╠═787215b4-c496-4484-9b86-16f3d3a7c432
# ╠═7e8ad8fd-20b8-4e25-9af5-1b582d487ec9
# ╠═108f4124-37d2-41c5-a3ce-c0d34ceae893
# ╠═63b8d015-f706-41a1-9242-e8df290257d6
# ╠═460d1857-9cf4-4cc6-a6ed-ccbc8301c609
# ╠═4c2e0a34-e66c-40ab-a84e-2bc35081f302
# ╠═22c7d696-50a5-47a4-a938-51745e80c5be
# ╠═cef73df7-da84-4105-9b41-33549ba00f6a
# ╠═c78f60e9-0a6c-4a75-98a0-df6094c7a70b
# ╠═35e9be7a-f171-4ac0-b7e5-c09dec2cf565
# ╠═a1f9711d-5dfa-4b90-b59b-36d808f36d15
# ╠═5f29d001-2fd6-4100-8214-7473b42a0221
# ╠═9f7e09b6-55fa-44ae-a5e9-9a5dd395f6ce
# ╠═b78e8829-dbbd-4afc-97d6-a93f40816f43
# ╠═92b23360-c00c-42c3-b515-1fced0d6d54b
# ╠═4630731e-22f6-4701-ba81-a59451e8c804
# ╠═c75844b7-8a6b-4482-b923-14aa97bac9fa
# ╠═6c06719a-7176-4e60-aceb-beb766007db2
# ╠═6527b895-6374-41eb-b4c1-ccdf96cec140
# ╠═c3d1e65e-5687-11f0-1275-d98935101b84
# ╠═31e371f6-1b02-42f7-b776-854e0aef651b
# ╠═330f6a43-394f-4c08-8b9f-5506cb9fd4c1
