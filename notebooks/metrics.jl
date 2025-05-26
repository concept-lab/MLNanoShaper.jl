### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 4dd0c84a-0b04-11f0-3528-37a16be3ea10
using Pkg;Pkg.activate(".")

# ╔═╡ 59084b85-b8d6-4fc6-86cd-9221eb3c9646
using TOML,MLUtils, Serialization, MarchingCubes, GeometryBasics,StructArrays, Folds, Statistics, NearestNeighbors

# ╔═╡ 86cba000-e329-4bd1-b290-23b88ad60b8a
import MLNanoShaper as MLN, MLNanoShaperRunner as MLNR, CairoMakie as Mk 

# ╔═╡ 4ef107ce-3c58-42fa-b4d2-67b4c016a8a6
parms = TOML.parsefile(MLN.params_file)

# ╔═╡ 853284a5-1be5-4f7d-83ab-2ebb50470049
trp = MLN.read_from_TOML(MLN.TrainingParameters,parms)

# ╔═╡ e4930118-ffd8-4087-9435-5eca6e764759
auxp = MLN.read_from_TOML(MLN.AuxiliaryParameters,parms)

# ╔═╡ 7e5c4473-4fc7-4d66-8ec9-4a0e2ee86858
(;test_data) = MLN.get_dataset(trp,auxp)

# ╔═╡ 60193da7-260c-484d-9696-bd35fd89d57f
sort(test_data.data.data[test_data.indices])

# ╔═╡ 448467e1-337f-4847-b42f-0221af4b0d30
model_weights = deserialize("$(homedir())/datasets/models/tiny_soft_max_angular_dense_s_test35_2025-05-23_epoch_500_773799609939854503")

# ╔═╡ 994a8229-d8f1-4048-912c-cb3d8d3c09dc
model = MLNR.production_instantiate(model_weights)

# ╔═╡ 757279cf-651e-4bc9-b487-b69bdb3b77f3
obs = getobs(test_data,1)

# ╔═╡ 79bb7396-0373-4334-a04a-d850057b3461
394*474*474/1000 /1000

# ╔═╡ 72f2eaca-1a28-43c7-99c9-7f463a553c93
function get_mesh(atoms::StructVector{<:Sphere},r::Float32=1f0)
	grid = MLNR.RegularGrid(atoms,3f0)
	mins = grid.start .- 2
	maxes = mins .+ size(grid.grid) .* grid.radius .+ 2
	x,y,z = map(1:3) do i collect(mins[i]:r:maxes[i]) end
	mc = MarchingCubes.MC(MLNR.evaluate_field(model,grid;step = r);x,y,z)
	march(mc,.5)
	MarchingCubes.makemesh(GeometryBasics, mc)
end

# ╔═╡ 3f7ca1a9-6683-476f-834c-dad92bda36ef


# ╔═╡ 2beaa16b-22e9-4c14-b34c-ec0f9f124338
r_grid =.5f0

# ╔═╡ c37bac7e-1624-4b97-8175-a1109bc4a91b
function has_neighborhood(p::Point3,r::MLNR.RegularGrid,radius::Number)::Bool
	r2 = radius^2
	!(Iterators.filter(x -> sum((p .- x).^2)<r2, MLNR._inrange(Vector{Point3f},r,p)) |> isempty)
end

# ╔═╡ ec06c92b-50aa-44a2-a1ba-5c4db56f5577
function precision(x::AbstractVector{Point3{T}},y::AbstractVector{Point3{T}};radius::Real)::T where T <: Real
	y_grid = MLNR.RegularGrid(y,T(1),identity)
	count(has_neighborhood.(x,Ref(y_grid),radius)) / length(x)
end

# ╔═╡ 1639c4f0-1d54-40c2-83a0-24c7468a4d2a
metrics = map(test_data[1:2])do (;atoms,skin)
	pred = get_mesh(atoms,r_grid)
	pred_coord = coordinates(pred)
	skin_coord = coordinates(skin)
	pred_tree = KDTree(pred_coord)
	skin_tree = KDTree(skin_coord)
	p = precision(pred_coord,skin_coord;radius=r_grid)
	r = precision(skin_coord,pred_coord;radius=r_grid)
	p_d = nn(skin_tree,pred_coord) |> last |> mean 
	r_d = nn(pred_tree,skin_coord) |> last |> mean 
	f = 2/(1/p +1/r)
	[p,r,f,p_d,r_d, p_d + r_d]
end |> stack

# ╔═╡ e12ed46f-de92-405d-93ef-39504d9c7374
mean(metrics;dims=2) |> vec

# ╔═╡ 2f75558c-89e6-4f36-9705-5983be5650e6
std(metrics;dims=2) |> vec

# ╔═╡ Cell order:
# ╠═4dd0c84a-0b04-11f0-3528-37a16be3ea10
# ╠═59084b85-b8d6-4fc6-86cd-9221eb3c9646
# ╠═86cba000-e329-4bd1-b290-23b88ad60b8a
# ╠═4ef107ce-3c58-42fa-b4d2-67b4c016a8a6
# ╠═853284a5-1be5-4f7d-83ab-2ebb50470049
# ╠═e4930118-ffd8-4087-9435-5eca6e764759
# ╠═7e5c4473-4fc7-4d66-8ec9-4a0e2ee86858
# ╠═60193da7-260c-484d-9696-bd35fd89d57f
# ╠═448467e1-337f-4847-b42f-0221af4b0d30
# ╠═994a8229-d8f1-4048-912c-cb3d8d3c09dc
# ╠═757279cf-651e-4bc9-b487-b69bdb3b77f3
# ╠═79bb7396-0373-4334-a04a-d850057b3461
# ╠═72f2eaca-1a28-43c7-99c9-7f463a553c93
# ╠═3f7ca1a9-6683-476f-834c-dad92bda36ef
# ╠═2beaa16b-22e9-4c14-b34c-ec0f9f124338
# ╠═1639c4f0-1d54-40c2-83a0-24c7468a4d2a
# ╠═e12ed46f-de92-405d-93ef-39504d9c7374
# ╠═2f75558c-89e6-4f36-9705-5983be5650e6
# ╠═c37bac7e-1624-4b97-8175-a1109bc4a91b
# ╠═ec06c92b-50aa-44a2-a1ba-5c4db56f5577
