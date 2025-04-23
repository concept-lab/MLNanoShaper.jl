### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 28210014-39eb-11ef-24eb-110acb81da08
using Pkg;Pkg.activate(".")

# ╔═╡ e9f0f433-0fe9-4096-b484-b432ec54afc8
using MLNanoShaper, MLNanoShaperRunner, FileIO, StructArrays, Static, Serialization,
      GeometryBasics, LuxCUDA, Lux, Profile, ProfileSVG, ChainRulesCore, Folds,
      BenchmarkTools, Zygote, Distances, LinearAlgebra, LoopVectorization, Folds,
      StaticTools, PProf, CUDA, Adapt, NearestNeighbors, MarchingCubes, FileIO, Transducers,Accessors, Revise, GeometryBasics, StaticArrays

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

# ╔═╡ c6e28039-0412-4327-a4a0-b4df80b5ef78
step = 1

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
@allocations MLNanoShaperRunner._inrange(Vector{Sphere},atoms,Point3f(10,22,0)) 

# ╔═╡ 862548ce-a24c-4396-a343-7a48e28ff11f
@benchmark _inrange(Matrix{Sphere{Float32}},atoms,b) 

# ╔═╡ 4c1d9e27-d23a-4817-ab9c-8c1a3142652b
x = [1,2]

# ╔═╡ 0a9217de-a955-4cad-81c6-86c9283caa01
b = Batch([Point3f(10,22,0) for _ in 1:1000])

# ╔═╡ fa13ff40-ff34-4a81-bd3e-74e0a10d5785
@allocations  _inrange(Matrix{Sphere{Float32}},atoms,b) 

# ╔═╡ a7700bc0-69e6-4f80-92ee-234aacce5bc6
@code_native _inrange(Matrix{Sphere{Float32}},atoms,b) 

# ╔═╡ d2390332-04a4-46c0-8ce5-0fdfb4179e03
@allocations MLNanoShaperRunner._inrange(Matrix{Sphere},atoms,Batch([Point3f(10,22,0) for _ in 1:20])) 

# ╔═╡ 08a75719-bb45-42e7-a544-1188a7c08e2b
@allocations Ref(1)

# ╔═╡ 29af96fc-56f9-48ed-9256-c88ad8c5416b
@benchmark MLNanoShaperRunner._inrange(Matrix{Sphere},atoms,Batch([Point3f(10,22,0) for _ in 1:1])) 

# ╔═╡ 688854e5-72c7-41eb-95de-86dab8f6b904
@benchmark MLNanoShaperRunner.__inrange(x ->nothing,atoms,Point3f(10,22,0),dx,dy,dz)

# ╔═╡ 017a44b2-f3e5-4760-8ba8-e907a04f81c0
dx = [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# ╔═╡ 2dc88a0b-73cb-4fde-afbd-acdb80477101
dy = [-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1]

# ╔═╡ 95128735-cb1d-4b15-a634-18c516d3fd3d
dz = [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1]

# ╔═╡ a10220cb-7235-4586-aa42-13f6cd2b5a96
function __inrange(f!::Function, g::RegularGrid{T}, p::Point3{T},dx::AbstractVector{Int},dy::AbstractVector{Int},dz::AbstractVector{Int}) where {T}
    index = get_id(p, g.start, g.radius)
	return
	x,y,z = index
    r2 = g.radius^2
    for i in 1:27
        x1 = x + dx[i]
        y1 = y + dy[i]
        z1 = z + dz[i]
        if x1 in axes(g.grid, 1) && y1 in axes(g.grid, 2) && z1 in axes(g.grid, 3)
            for s in g.grid[x1, y1, z1]
                if sum((p .- g.center(s)) .^ 2) < r2
                    nothing
                end
            end
        end
    end
end

# ╔═╡ e215e7dc-8fbd-45cf-a750-789c49c41029
function _inrange(::Type{G}, g::RegularGrid{T}, p::Batch{<:AbstractVector{Point3{T}}}) where {T,G}
    n = length(p.field)
    res = MLNanoShaperRunner._summon_type(G)(undef, 128, n)
    ret = Vector{SubArray{eltype(G),1,G,Tuple{UnitRange{Int64},Int},true}}(undef, n)
	i = Ref(1)
    for j in 1:n
		i[] = 1
        __inrange(x -> MLNanoShaperRunner.my_push!(res,i,j,x), g, p.field[j],dx,dy,dz)
		#@info typeof(@view res[1:i[],j])
        ret[j] =@view res[1:i[],j]
    end
    ret
end

# ╔═╡ dea65c13-ddac-4c26-a388-b6b931533782
@allocations __inrange(x ->nothing,atoms,Point3f(-1.84,15.696,-3.1),dx,dy,dz)

# ╔═╡ d38d4fca-eacd-476d-8f96-fe71e81f8b45
@code_native __inrange(x ->nothing,atoms,Point3f(-1.84,15.696,-3.1),dx,dy,dz)

# ╔═╡ efc64b46-6795-4395-9e03-ac81ce7d10c4
@inline function get_id(point::Point3{T}, start::Point3{T}, radius::T)::Point3{Int} where T
    scaled_offset = (point .- start) ./ radius
    return unsafe_trunc.(Int64, scaled_offset ) .+ 1
end

# ╔═╡ 9579140a-0795-447e-9880-65fee4511579
trunc(Int64,-1.1)

# ╔═╡ 722045b9-9522-4fc3-8333-546e438ea1a1
floor(Int64,-1.1)

# ╔═╡ 8281213a-ec6b-43c6-b5ef-bc3781489dbf
@code_native atoms.center

# ╔═╡ bc1a6493-4f30-4080-8f90-736290edab7c
atoms.grid |> typeof

# ╔═╡ e66cfea3-cc28-4e5f-9054-52132c0f0714
typeof(atoms)

# ╔═╡ 402ca037-bbff-4917-88ab-49455da9d98c
function f()
	p = Point3f(-1.84,15.696,-3.1)
	g = atoms
	coord =  get_id(p, g.start, g.radius)
	x,y,z = coord
    r2 = g.radius^2
    for i in 1:27
        x1 = x + dx[i]
        y1 = y + dy[i]
        z1 = z + dz[i]
        if x1 in axes(g.grid, 1) && y1 in axes(g.grid, 2) && z1 in axes(g.grid, 3)
            for s in g.grid[x1, y1, z1]
                if sum((p .- g.center(s)) .^ 2) < r2
                    nothing
                end
            end
        end
    end
end

# ╔═╡ f20b112f-a01b-4da2-b5ba-a7b37b6b944f
@allocated MLNanoShaperRunner.get_id(Point3f(-1.84,15.696,-3.1), atoms.start, atoms.radius)

# ╔═╡ 6de0dc85-76da-405b-8bfb-d0d8f66e3d58
atoms.grid[8,8,5][1]

# ╔═╡ f2343acb-23c2-4155-b393-c0bcea4d9760
#@benchmark model((MLNanoShaperRunner.Batch([Point3f(10,22,0)]),atoms))

# ╔═╡ 187ca178-6b4c-406b-b312-81e12026b720
#@benchmark model((MLNanoShaperRunner.Batch([Point3f(10,22,0),Point3f(10,22,0)]),atoms))

# ╔═╡ f49f8d87-540b-4767-bbb2-794cab2da54a
#@benchmark model((MLNanoShaperRunner.Batch([Point3f(10,22,0) for _ in 1:1000]),atoms))

# ╔═╡ f723350c-c5c2-428b-9617-4db6722d87f7
#@benchmark model((MLNanoShaperRunner.Batch([Point3f(10,22,0) for _ in 1:100000]),atoms))

# ╔═╡ cb7a84e8-d5ae-4f43-a1ca-bd047f39dd2b
#@benchmark model_gpu((MLNanoShaperRunner.Batch([Point3f(10,22,0) for _ in 1:100000]),atoms)

# ╔═╡ fac62889-166c-43dc-8bb9-210f217ebcb0
#@benchmark model.model.layers[1].fun((MLNanoShaperRunner.Batch([Point3f(10,22,0) for _ in 1:100000]),atoms))

# ╔═╡ 1ba4d717-e948-4a8e-b96b-0bc21d284e54
(11 - 9.5)/ (77 - 9.5 )

# ╔═╡ c97eb631-f5fa-4f13-a7bf-2bc874c527d2
#CUDA.@profile model_gpu((MLNanoShaperRunner.Batch([Point3f(10,22,0) for _ in 1:1000]),atoms))

# ╔═╡ d634d86e-3db5-4d25-8177-0fcb7ff18643
77/20

# ╔═╡ bb374966-08bf-420b-a486-eba42ad359ce
#@benchmark model.model.layers[1].fun((MLNanoShaperRunner.Batch([Point3f(10,22,0) for _ in 1:10]),atoms))

# ╔═╡ 376569b8-1225-4b44-9eae-62bdba87eed1
#@benchmark MLNanoShaperRunner.select_neighboord(Point3f(10,22,0),atoms)

# ╔═╡ 387b3778-a710-4bba-9942-b6119d1561da
length(atoms.grid) * 2e-6 * 3*3*3 / 12

# ╔═╡ b4cb025a-e473-4feb-aace-a533503c3672
model_weights = deserialize("$(homedir())/datasets/models/tiny_angular_dense_s_jobs_14_6_3_c_2025-03-19_epoch_400_9592899277305186470")

# ╔═╡ a820838f-7105-4770-8a26-b4cb4af3bec1
model_gpu = MLNanoShaperRunner.production_instantiate(model_weights,on_gpu=true)

# ╔═╡ e6f8e419-2fb4-4c8a-afd3-05e500553cfc
model_fixed_size = Lux.StatefulLuxLayer{true}(model_weights.model(max_nb_atoms = 10),model_weights.parameters,model_weights.states)

# ╔═╡ 5e6d15a4-c551-40bd-97ed-57c787734217
gpu_device()

# ╔═╡ 0adf29e7-a6e4-48ae-bfe0-e5340d1d1a70
model = MLNanoShaperRunner.production_instantiate(model_weights)

# ╔═╡ 0d81f6cb-c6d4-4748-b23c-46ab1b0f9a8e
cutoff_radius = 3.0f0

# ╔═╡ 2ff3238a-2205-405a-9075-553f27db84d6


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
    @profile [MLNanoShaperRunner._inrange(Matrix{Sphere},atoms,Batch([Point3f(10,22,0) for _ in 1:100])) for _ in 1:100] 
	pprof()
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═e4fc0299-2b72-4b8f-940d-9f55a76f83ca
# ╠═28210014-39eb-11ef-24eb-110acb81da08
# ╠═e9f0f433-0fe9-4096-b484-b432ec54afc8
# ╠═e4a81477-34da-4891-9d0e-34a30ada4ac3
# ╠═c6e28039-0412-4327-a4a0-b4df80b5ef78
# ╠═47641b85-0596-4ce4-992b-6811ff89574b
# ╠═5765cbc5-ec12-406e-b43f-9291c99b9d1d
# ╠═2cce8a1a-97fe-45ae-bca7-584b843739d6
# ╠═d9898910-1b29-400c-bcea-457017723c70
# ╠═08abee3c-49ee-42a1-adae-7b5a7d09a8f5
# ╠═e215e7dc-8fbd-45cf-a750-789c49c41029
# ╠═862548ce-a24c-4396-a343-7a48e28ff11f
# ╠═4c1d9e27-d23a-4817-ab9c-8c1a3142652b
# ╠═0a9217de-a955-4cad-81c6-86c9283caa01
# ╠═fa13ff40-ff34-4a81-bd3e-74e0a10d5785
# ╠═a7700bc0-69e6-4f80-92ee-234aacce5bc6
# ╠═d2390332-04a4-46c0-8ce5-0fdfb4179e03
# ╠═08a75719-bb45-42e7-a544-1188a7c08e2b
# ╠═29af96fc-56f9-48ed-9256-c88ad8c5416b
# ╠═688854e5-72c7-41eb-95de-86dab8f6b904
# ╠═017a44b2-f3e5-4760-8ba8-e907a04f81c0
# ╠═2dc88a0b-73cb-4fde-afbd-acdb80477101
# ╠═95128735-cb1d-4b15-a634-18c516d3fd3d
# ╠═a10220cb-7235-4586-aa42-13f6cd2b5a96
# ╠═dea65c13-ddac-4c26-a388-b6b931533782
# ╠═d38d4fca-eacd-476d-8f96-fe71e81f8b45
# ╠═efc64b46-6795-4395-9e03-ac81ce7d10c4
# ╠═9579140a-0795-447e-9880-65fee4511579
# ╠═722045b9-9522-4fc3-8333-546e438ea1a1
# ╠═8281213a-ec6b-43c6-b5ef-bc3781489dbf
# ╠═bc1a6493-4f30-4080-8f90-736290edab7c
# ╠═e66cfea3-cc28-4e5f-9054-52132c0f0714
# ╠═402ca037-bbff-4917-88ab-49455da9d98c
# ╠═f20b112f-a01b-4da2-b5ba-a7b37b6b944f
# ╠═6de0dc85-76da-405b-8bfb-d0d8f66e3d58
# ╠═f2343acb-23c2-4155-b393-c0bcea4d9760
# ╠═187ca178-6b4c-406b-b312-81e12026b720
# ╠═f49f8d87-540b-4767-bbb2-794cab2da54a
# ╠═f723350c-c5c2-428b-9617-4db6722d87f7
# ╠═cb7a84e8-d5ae-4f43-a1ca-bd047f39dd2b
# ╠═fac62889-166c-43dc-8bb9-210f217ebcb0
# ╠═1ba4d717-e948-4a8e-b96b-0bc21d284e54
# ╠═c97eb631-f5fa-4f13-a7bf-2bc874c527d2
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
# ╠═f44f276c-7de9-49dc-b584-5a64a04006a0
# ╠═70663eda-5bd3-4f08-8792-5f848edccaff
# ╠═1b16179f-8da8-4c34-9455-5554a3151f40
# ╠═988caa2b-a9cf-4226-beb2-52efa750beca
# ╠═565892fa-df36-4fad-8872-a9e2f7de684f
# ╠═34d53b3e-0f9e-4088-aa7d-b1acf8516e4b
# ╠═5273eaf5-2762-41ba-b634-17e170adc65e
