### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 4dd0c84a-0b04-11f0-3528-37a16be3ea10
using Pkg;Pkg.activate(".")

# ╔═╡ 703cd041-33f7-41ce-9e25-aae874e40ed7
using Revise

# ╔═╡ 59084b85-b8d6-4fc6-86cd-9221eb3c9646
using TOML,MLUtils, Serialization, MarchingCubes, GeometryBasics,StructArrays, Folds, Statistics, NearestNeighbors, DataFrames, CSV, FileIO, PDBTools,Lux

# ╔═╡ 86cba000-e329-4bd1-b290-23b88ad60b8a
import MLNanoShaper as MLN, MLNanoShaperRunner as MLNR, CairoMakie as Mk 

# ╔═╡ 4ef107ce-3c58-42fa-b4d2-67b4c016a8a6
parms = TOML.parsefile(MLN.params_file)

# ╔═╡ 4ca3c279-9180-4a05-bf94-66a8def52310
function radius(x::AbstractString)::Float32
	if x in keys(parms["atoms"]["radius"])
		parms["atoms"]["radius"][x]
	else
		1f0
	end
end

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
	"tiny_angular_dense_s_final_training_10_3.0_categorical_6000_6331735514142882335",
	"tiny_angular_dense_s_final_training_10_3.0_continuous_6000_6930415729134250803",
	"light_angular_dense_s_final_training_10_3.0_categorical_3000_13233356928103905208",
	"light_angular_dense_s_final_training_10_3.0_continuous_3000_16971193858460515576",
	"tiny_angular_dense_s_final_training_10_4.0_categorical_3000_7858487215662934347",
	"tiny_angular_dense_s_final_training_10_4.0_continuous_3000_17513372773368690140",
	"light_angular_dense_s_final_training_10_4.0_categorical_3000_7024144549625149892",
	"light_angular_dense_s_final_training_10_4.0_continuous_3000_4435365212306114114"
]

# ╔═╡ b18c5fde-6a25-492f-a754-a2c45e3f2eec
names = ["tiny categorical 3Å","tiny continuous 3Å","light categorical 3Å","light continuous 3Å", "tiny categorical 4Å","tiny continuous 4Å","light categorical 4Å","light continuous 4Å"]

# ╔═╡ 448467e1-337f-4847-b42f-0221af4b0d30
model_weights = deserialize("$(homedir())/datasets/models/tiny_soft_max_angular_dense_testhardsigma3_35000_933236481126930411")

# ╔═╡ 994a8229-d8f1-4048-912c-cb3d8d3c09dc
model = MLNR.production_instantiate(model_weights;on_gpu=true)

# ╔═╡ 92118d0c-75b9-4035-92a0-0c836cc27c89
models = MLNR.production_instantiate.(map(p -> "$(homedir())/datasets/models/$p",models_paths) .|> deserialize,on_gpu=true)

# ╔═╡ 8fcd0922-dc70-42aa-95d0-a4f3d4d0455b
function make_pdb_dataset(path::AbstractString)
	all_files = readdir(path)
	pdb_files = filter(file -> endswith(file, ".pdb"), all_files)
	mapobs(pdb_files) do file
		cd(path) do
			atoms = map(read_pdb(file)) do (;x,y,z,occup)
				Sphere(Point3f(x,y,z),occup)
			end |> StructVector 
			skin = "$(splitext(file) |> first).off" |> load
			(;atoms,skin)
		end
	end
end

# ╔═╡ 757279cf-651e-4bc9-b487-b69bdb3b77f3
obs = getobs(test_data,1)

# ╔═╡ 72f2eaca-1a28-43c7-99c9-7f463a553c93
function get_mesh_and_field(model,atoms::StructVector{<:Sphere},atoms_grid::MLNR.RegularGrid,r::Float32=1f0)
	mins = atoms_grid.start .- 2
	maxes = (atoms_grid.start .+ size(atoms_grid.grid) .* atoms_grid.radius) .+ 2
	x,y,z = map(1:3) do i collect(mins[i]:r:maxes[i]) end
	field = MLNR.evaluate_field_fast(model,atoms;step = r,batch_size=10000)
	mc = MarchingCubes.MC(field;x,y,z)
	march(mc,.5)
	MarchingCubes.makemesh(GeometryBasics, mc),field
end

# ╔═╡ 8e6d1f03-a240-4c84-a820-b8ce58dcd316
function get_grid_points(start,en,step)
	mins = start .- 2
	maxes = en .+ 2
    ranges = range.(mins, maxes; step)
	Point3f.(reshape(ranges[1], :, 1,1), reshape(ranges[2], 1, :,1), reshape(ranges[3], 1,1,:))
end

# ╔═╡ 2beaa16b-22e9-4c14-b34c-ec0f9f124338
r_grid =.5f0

# ╔═╡ 972048bf-5ad6-4551-a3b4-df70b09c1925
MAE(X,Y) = mean(abs.(X .- Y))

# ╔═╡ 4d5c05fb-88b0-4b21-83a9-5ca0952d0032
SIE(X,Y) = mean(sign.( X .- .5) .!= sign.( Y .- .5))

# ╔═╡ dcd6c2ac-2554-4a19-ba52-869daaa92e61
SIE(1,1)

# ╔═╡ 9acac13b-7d07-4b2d-b2a8-54289e528134
categorical_value(distance) = ifelse(distance > 0,1f0,ifelse(distance < 0 ,0f0,.5f0))

# ╔═╡ 52bbeee7-480b-4545-8720-8af6748f7344
continuous_value(distance) = 1 /(1 +exp( - distance /.05f0)) 

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
function get_dataframe_metrics(models::AbstractVector{<:Lux.StatefulLuxLayer},names,data,loss_functions)
	lines = map(zip(models,loss_functions)) do (model,loss_f)
		cutoff_radius = MLNR.get_cutoff_radius(model.model)
		metrics = map(data)do (;atoms,skin)
			grid = MLNR.RegularGrid(atoms,cutoff_radius)
			pred,pred_field = get_mesh_and_field(model,atoms,grid,r_grid)
			pred_coord = coordinates(pred)
			skin_coord = coordinates(skin)
			pred_tree = KDTree(pred_coord)
			skin_tree = KDTree(skin_coord)
			dist_field = MLN.signed_distance.(get_grid_points(grid.start,grid.start .+ size(grid.grid) .* grid.radius,r_grid), Ref(MLN.RegionMesh(skin))) |> gpu_device() .|> loss_f |> cpu_device()
			mae = MAE(pred_field,dist_field)
			sie = SIE(pred_field,dist_field)
			p = precision(pred_coord,skin_coord;radius=r_grid)
			r = precision(skin_coord,pred_coord;radius=r_grid)
			r_d = nn(skin_tree,pred_coord) |> last |> mean 
			p_d = nn(pred_tree,skin_coord) |> last |> mean 
			f = 2/(1/p +1/r)
			[p,r,f,p_d,r_d, p_d + r_d,mae,sie]
		end |> stack
		mean(metrics;dims=2) |> vec
	end
	hcat(DataFrame(;model=names),DataFrame(stack(lines;dims=1),["precision","recall","fscore","distance_to_predicted","distance_to_true","CD","MAE","SIE"]))
end

# ╔═╡ 8ba466ba-1d9f-4a20-a5cb-a12e5a2200ac
nucleic_path = "$(homedir())/workspace/Con2SES/datasets/nucleic_acid_test"

# ╔═╡ fcb0b00f-f3d7-48cd-9de2-1ade87f0b0ac
protein_complex_path = "$(homedir())/workspace/Con2SES/datasets/protein_complex_test"

# ╔═╡ be736578-6b9a-424e-b019-fbc6086a9aa4
nucleic_dataset = first(make_pdb_dataset(nucleic_path),2)

# ╔═╡ cfe27201-186c-4a36-923b-57543e7b4d02
protein_complex_dataset = make_pdb_dataset(protein_complex_path)

# ╔═╡ 8a3ae0f5-f587-4db6-b82f-42751546534d
loss_functions = [categorical_value,continuous_value,categorical_value,continuous_value,categorical_value,continuous_value,categorical_value,continuous_value,]

# ╔═╡ 187cd2eb-7630-4a1b-a020-2e75a4f3ddce
df = get_dataframe_metrics(models,names,test_data,loss_functions)

# ╔═╡ d87429da-f6f4-4b4f-9ea7-af6ca226f532
CSV.write("metrics.csv",df)

# ╔═╡ c85986dc-51e7-41eb-b351-0c407dbccf86
df_nucleic = get_dataframe_metrics(models,names,nucleic_dataset,loss_functions)

# ╔═╡ b557e9bf-923c-404f-a268-c2bc443d36a0
#df_protein_complex = get_dataframe_metrics(models,names,protein_complex_dataset,loss_functions) 

# ╔═╡ 6e45b065-569a-4603-a87f-d224c79c3d1c
CSV.write("metrics_nucleics.csv",df_nucleic)

# ╔═╡ 8f97bff6-4486-4419-821e-cc7ef12b7d90
#CSV.write("metrics_protein_complex.csv",df_protein_complex)

# ╔═╡ Cell order:
# ╠═4dd0c84a-0b04-11f0-3528-37a16be3ea10
# ╠═703cd041-33f7-41ce-9e25-aae874e40ed7
# ╠═59084b85-b8d6-4fc6-86cd-9221eb3c9646
# ╠═86cba000-e329-4bd1-b290-23b88ad60b8a
# ╠═4ef107ce-3c58-42fa-b4d2-67b4c016a8a6
# ╠═4ca3c279-9180-4a05-bf94-66a8def52310
# ╠═853284a5-1be5-4f7d-83ab-2ebb50470049
# ╠═e4930118-ffd8-4087-9435-5eca6e764759
# ╠═7e5c4473-4fc7-4d66-8ec9-4a0e2ee86858
# ╠═60193da7-260c-484d-9696-bd35fd89d57f
# ╠═1d338fd4-6c87-4829-8942-cf97d0db0aad
# ╠═b18c5fde-6a25-492f-a754-a2c45e3f2eec
# ╠═448467e1-337f-4847-b42f-0221af4b0d30
# ╠═994a8229-d8f1-4048-912c-cb3d8d3c09dc
# ╠═92118d0c-75b9-4035-92a0-0c836cc27c89
# ╠═8fcd0922-dc70-42aa-95d0-a4f3d4d0455b
# ╠═757279cf-651e-4bc9-b487-b69bdb3b77f3
# ╠═72f2eaca-1a28-43c7-99c9-7f463a553c93
# ╠═8e6d1f03-a240-4c84-a820-b8ce58dcd316
# ╠═2beaa16b-22e9-4c14-b34c-ec0f9f124338
# ╠═972048bf-5ad6-4551-a3b4-df70b09c1925
# ╠═4d5c05fb-88b0-4b21-83a9-5ca0952d0032
# ╠═dcd6c2ac-2554-4a19-ba52-869daaa92e61
# ╠═9acac13b-7d07-4b2d-b2a8-54289e528134
# ╠═52bbeee7-480b-4545-8720-8af6748f7344
# ╠═957f510f-54e2-4d6e-81b3-022de69702e0
# ╠═c37bac7e-1624-4b97-8175-a1109bc4a91b
# ╠═ec06c92b-50aa-44a2-a1ba-5c4db56f5577
# ╠═187cd2eb-7630-4a1b-a020-2e75a4f3ddce
# ╠═d87429da-f6f4-4b4f-9ea7-af6ca226f532
# ╠═8ba466ba-1d9f-4a20-a5cb-a12e5a2200ac
# ╠═fcb0b00f-f3d7-48cd-9de2-1ade87f0b0ac
# ╠═be736578-6b9a-424e-b019-fbc6086a9aa4
# ╠═cfe27201-186c-4a36-923b-57543e7b4d02
# ╠═8a3ae0f5-f587-4db6-b82f-42751546534d
# ╠═c85986dc-51e7-41eb-b351-0c407dbccf86
# ╠═b557e9bf-923c-404f-a268-c2bc443d36a0
# ╠═6e45b065-569a-4603-a87f-d224c79c3d1c
# ╠═8f97bff6-4486-4419-821e-cc7ef12b7d90
