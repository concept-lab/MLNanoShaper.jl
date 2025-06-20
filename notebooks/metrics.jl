### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 4dd0c84a-0b04-11f0-3528-37a16be3ea10
using Pkg;Pkg.activate(".")

# ╔═╡ 59084b85-b8d6-4fc6-86cd-9221eb3c9646
using TOML,MLUtils, Serialization, MarchingCubes, GeometryBasics,StructArrays, Folds, Statistics, NearestNeighbors, DataFrames, CSV

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

# ╔═╡ 1d338fd4-6c87-4829-8942-cf97d0db0aad
models_paths = String[
	"tiny_soft_max_angular_dense_jobs_40_3_2025-06-04_epoch_4370_75456560920357034",
	"light_soft_max_angular_dense_jobs_40_3_2025-06-04_epoch_2870_2744795346301154981",
	"light_soft_max_angular_dense_jobs_40_4_2025-05-31_epoch_560_1951733446584503143",
	"light_soft_max_angular_dense_jobs_40_5d_2025-06-04_epoch_120_8715503880527416838",
]

# ╔═╡ 448467e1-337f-4847-b42f-0221af4b0d30
model_weights = deserialize("$(homedir())/datasets/models/tiny_soft_max_angular_dense_jobs_40_3_2025-06-04_epoch_4370_75456560920357034")

# ╔═╡ 994a8229-d8f1-4048-912c-cb3d8d3c09dc
model = MLNR.production_instantiate(model_weights;on_gpu=true)

# ╔═╡ 92118d0c-75b9-4035-92a0-0c836cc27c89
models = MLNR.production_instantiate.(map(p -> "$(homedir())/datasets/models/$p",models_paths) .|> deserialize,on_gpu=true) 

# ╔═╡ 757279cf-651e-4bc9-b487-b69bdb3b77f3
obs = getobs(test_data,1)

# ╔═╡ 72f2eaca-1a28-43c7-99c9-7f463a553c93
function get_mesh(model,atoms::StructVector{<:Sphere},r::Float32=1f0)
	grid = MLNR.RegularGrid(atoms,3f0)
	mins = grid.start .- 2
	maxes = mins .+ size(grid.grid) .* grid.radius .+ 2
	x,y,z = map(1:3) do i collect(mins[i]:r:maxes[i]) end
	mc = MarchingCubes.MC(MLNR.evaluate_field(model,grid;step = r);x,y,z)
	march(mc,.5)
	MarchingCubes.makemesh(GeometryBasics, mc)
end

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

# ╔═╡ 957f510f-54e2-4d6e-81b3-022de69702e0
function get_dataframe_metrics(models,names)
	lines = map(models) do model
		metrics = map(test_data[1:20])do (;atoms,skin)
			pred = get_mesh(model,atoms,r_grid)
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
		mean(metrics;dims=2) |> vec
	end
	hcat(DataFrame(;model=names),DataFrame(stack(lines;dims=1),["precision","recall","fscore","distance_to_predicted","distance_to_true","CD"]))
end

# ╔═╡ 187cd2eb-7630-4a1b-a020-2e75a4f3ddce
df = get_dataframe_metrics(models,["tiny softmax 3A","light softmax 3A", "light softmax 4A","light softmax 5A"]) 

# ╔═╡ d87429da-f6f4-4b4f-9ea7-af6ca226f532
CSV.write("metrics.csv",df)

# ╔═╡ Cell order:
# ╠═4dd0c84a-0b04-11f0-3528-37a16be3ea10
# ╠═59084b85-b8d6-4fc6-86cd-9221eb3c9646
# ╠═86cba000-e329-4bd1-b290-23b88ad60b8a
# ╠═4ef107ce-3c58-42fa-b4d2-67b4c016a8a6
# ╠═853284a5-1be5-4f7d-83ab-2ebb50470049
# ╠═e4930118-ffd8-4087-9435-5eca6e764759
# ╠═7e5c4473-4fc7-4d66-8ec9-4a0e2ee86858
# ╠═60193da7-260c-484d-9696-bd35fd89d57f
# ╠═1d338fd4-6c87-4829-8942-cf97d0db0aad
# ╠═448467e1-337f-4847-b42f-0221af4b0d30
# ╠═994a8229-d8f1-4048-912c-cb3d8d3c09dc
# ╠═92118d0c-75b9-4035-92a0-0c836cc27c89
# ╠═757279cf-651e-4bc9-b487-b69bdb3b77f3
# ╠═72f2eaca-1a28-43c7-99c9-7f463a553c93
# ╠═2beaa16b-22e9-4c14-b34c-ec0f9f124338
# ╠═957f510f-54e2-4d6e-81b3-022de69702e0
# ╠═187cd2eb-7630-4a1b-a020-2e75a4f3ddce
# ╠═d87429da-f6f4-4b4f-9ea7-af6ca226f532
# ╠═c37bac7e-1624-4b97-8175-a1109bc4a91b
# ╠═ec06c92b-50aa-44a2-a1ba-5c4db56f5577
