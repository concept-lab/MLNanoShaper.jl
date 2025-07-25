### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 23ec82f2-5d85-11f0-0ff0-cbf9fa44e401
using Pkg;Pkg.activate(".") 

# ╔═╡ 980a41a3-f40e-40bd-aea6-2f60c437a2b5
using Revise

# ╔═╡ 5af29509-ec0c-4bba-a745-45b8b19e615e
using MLNanoShaper, MLNanoShaperRunner, FileIO, StructArrays, Static, Serialization,
      GeometryBasics, LuxCUDA, Lux, Profile, ProfileSVG, ChainRulesCore, Folds,
      BenchmarkTools, Zygote, Distances, LinearAlgebra, LoopVectorization, Folds,
      StaticTools, PProf, CUDA, Adapt, NearestNeighbors, MarchingCubes, FileIO, Transducers,Accessors, TOML, DataFrames

# ╔═╡ d1d70b52-aaf2-4e90-87dc-bd8f09e94f00
import CairoMakie as Mk

# ╔═╡ 3a650977-0a52-448f-88cb-8cebd67886bc
model_name ="tiny_angular_dense_s_final_training_10_3.0_categorical_6000_6331735514142882335"

# ╔═╡ 04a6466c-4c5b-49d8-a4b9-30e9b57e4bc6
model_weights = deserialize("$(homedir())/datasets/models/$model_name")

# ╔═╡ 81b5b6a4-c90f-4a9d-82ab-1b89d448e360
model = MLNanoShaperRunner.production_instantiate(model_weights,on_gpu=true) 

# ╔═╡ 1918aaec-d7af-4e19-be8e-26c0cb7bf70f
prot_num=2

# ╔═╡ 8f8af861-70a5-46d3-83fc-1b9fc970dc16
vec_atoms = getfield.(
        read("$(homedir())/datasets/pqr/$prot_num/structure.pqr", PQR{Float32}), :pos) |>
    StructVector

# ╔═╡ 9357cf2b-4362-4cf7-9238-087d7f1fc7dd
length(vec_atoms)

# ╔═╡ c5d6741d-7b17-4eb9-b9cf-304c61b5fe83
function get_time_point(fun,prot_num::Number)
	vec_atoms = getfield.(
        read("$(homedir())/datasets/pqr/$prot_num/structure.pqr", PQR{Float32}), :pos) |>
    StructVector 
	k= fun(vec_atoms)
	length(vec_atoms),(k.times |> median) * 1e-9
end

# ╔═╡ 69d8b38e-1526-4c27-9705-af696eaad5d3
get_nb_lines(id) = getfield.(
        read("$(homedir())/datasets/pqr/$id/structure.pqr", PQR{Float32}), :pos) |> StructVector |> length

# ╔═╡ db4627bf-53fa-4dd3-83cd-7143c76a1cc9
parms = TOML.parsefile(MLNanoShaper.params_file)

# ╔═╡ 463d282e-b770-441e-8d0b-81d1ea9f2f24
trp = MLNanoShaper.read_from_TOML(MLNanoShaper.TrainingParameters,parms)

# ╔═╡ 4efe8a36-df45-47d4-88aa-c80c171c74b8
auxp = MLNanoShaper.read_from_TOML(MLNanoShaper.AuxiliaryParameters,parms)

# ╔═╡ 54bcde7c-23f1-4d5f-a867-89e8699ac6cf
(;test_data,train_data) = MLNanoShaper.get_dataset(trp,auxp)

# ╔═╡ 1cb847ed-c204-4274-81ea-3f5311029118
ids = test_data.data.data[test_data.indices] |> sort

# ╔═╡ 65579214-b7a7-4339-82a2-8464146bab33
g(a) = @benchmark MLNanoShaperRunner.evaluate_field_fast(model,$a;step=.5f0)

# ╔═╡ b163e4d2-0b12-4fef-b0cd-5da80434d547
get_time_point(g,19) 

# ╔═╡ aa9d7742-e5fe-42bc-a6c7-516aee8ce0a5
get_time_point(g,50) 

# ╔═╡ d42788b4-92f2-493a-b01b-c570fe47c790
get_nb_lines(19)

# ╔═╡ f7f641c1-79a7-4d98-94bc-47fc49036407
get_nb_lines(50)

# ╔═╡ af63153d-db6b-4db8-aeef-f78e1d267436
get_nb_lines.(ids) |>argmax

# ╔═╡ d1c44c55-7a8f-46fd-aa96-818a155e2ad0
ids[6]

# ╔═╡ 3d4719b4-bd6d-4bbf-934a-77c26df806f7
cdtempdir(f,args...;kargs...) =  mktempdir(args...;kargs...) do dir
	cd(f,dir) 
end

# ╔═╡ ae629a79-92d9-4bd6-b4cb-a245a8ca934f
function write_pqr(io::IO,atoms::AbstractVector{Sphere{Float32}})
	for (;center,r) in atoms
		x,y,z = center
		println(io,x," ",y," ",z," ",r)
	end
end

# ╔═╡ ea19b150-f555-40cd-9715-6b92af6b6424
function run_nanoshaper(atoms::AbstractVector{Sphere{Float32}})
	cdtempdir() do
        open("structure.xyzr","w") do io
			write_pqr(io,atoms)
		end
			
		conf_path = joinpath(@__DIR__, "conf.prm")
        symlink(conf_path, "conf.prm")
        command = addenv(
			`$(homedir())/workspace/nanoshaper/pkg_nanoshaper_0.7.8/NanoShaper conf.prm`,
			"LD_LIBRARY_PATH" => "$(homedir())/workspace/nanoshaper/pkg_nanoshaper_0.7.8")
		@benchmark run(pipeline($command,devnull);wait = true)
	end
end

# ╔═╡ d0b0c9e0-da1c-4458-989a-faf79c249b3e
get_time_point(run_nanoshaper,19) 

# ╔═╡ 5d7fc4d8-9843-40aa-a06b-126c26753e26
get_time_point(run_nanoshaper,50)

# ╔═╡ 49779c1f-fe10-4855-9da4-3e0d79d438e8
rev_vals =  get_time_point.(run_nanoshaper,ids)

# ╔═╡ 524ccba2-af86-4207-bdf7-fbf56ae01017
vals  = get_time_point.(g,ids) 

# ╔═╡ 4e622491-9874-4ef1-a4cf-33a385fc38b9
begin
	f = Mk.Figure()
	ax = Mk.Axis(f[1,1],xlabel="nb atoms",ylabel = "execution time(s)")
	Mk.plot!(ax,vals,color=:blue,label="MLNanoShaper")
	Mk.plot!(ax,rev_vals,color=:red,label="nanoShaper")
	f[1,2] = Mk.Legend(f,ax)
	f
end

# ╔═╡ bda67f44-273a-4e67-8d45-864d8f2bcb51
save("execution_time.pdf",f)

# ╔═╡ Cell order:
# ╠═23ec82f2-5d85-11f0-0ff0-cbf9fa44e401
# ╠═980a41a3-f40e-40bd-aea6-2f60c437a2b5
# ╠═d1d70b52-aaf2-4e90-87dc-bd8f09e94f00
# ╠═5af29509-ec0c-4bba-a745-45b8b19e615e
# ╠═3a650977-0a52-448f-88cb-8cebd67886bc
# ╠═04a6466c-4c5b-49d8-a4b9-30e9b57e4bc6
# ╠═81b5b6a4-c90f-4a9d-82ab-1b89d448e360
# ╠═1918aaec-d7af-4e19-be8e-26c0cb7bf70f
# ╠═8f8af861-70a5-46d3-83fc-1b9fc970dc16
# ╠═9357cf2b-4362-4cf7-9238-087d7f1fc7dd
# ╠═c5d6741d-7b17-4eb9-b9cf-304c61b5fe83
# ╠═69d8b38e-1526-4c27-9705-af696eaad5d3
# ╠═db4627bf-53fa-4dd3-83cd-7143c76a1cc9
# ╠═463d282e-b770-441e-8d0b-81d1ea9f2f24
# ╠═4efe8a36-df45-47d4-88aa-c80c171c74b8
# ╠═54bcde7c-23f1-4d5f-a867-89e8699ac6cf
# ╠═1cb847ed-c204-4274-81ea-3f5311029118
# ╠═65579214-b7a7-4339-82a2-8464146bab33
# ╠═b163e4d2-0b12-4fef-b0cd-5da80434d547
# ╠═d0b0c9e0-da1c-4458-989a-faf79c249b3e
# ╠═aa9d7742-e5fe-42bc-a6c7-516aee8ce0a5
# ╠═5d7fc4d8-9843-40aa-a06b-126c26753e26
# ╠═d42788b4-92f2-493a-b01b-c570fe47c790
# ╠═f7f641c1-79a7-4d98-94bc-47fc49036407
# ╠═af63153d-db6b-4db8-aeef-f78e1d267436
# ╠═d1c44c55-7a8f-46fd-aa96-818a155e2ad0
# ╠═ea19b150-f555-40cd-9715-6b92af6b6424
# ╠═3d4719b4-bd6d-4bbf-934a-77c26df806f7
# ╠═ae629a79-92d9-4bd6-b4cb-a245a8ca934f
# ╠═49779c1f-fe10-4855-9da4-3e0d79d438e8
# ╠═524ccba2-af86-4207-bdf7-fbf56ae01017
# ╠═4e622491-9874-4ef1-a4cf-33a385fc38b9
# ╠═bda67f44-273a-4e67-8d45-864d8f2bcb51
