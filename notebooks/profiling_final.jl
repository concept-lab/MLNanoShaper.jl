### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ b4c8f01b-5459-4a60-9a43-e85f6b2c05ee
using Pkg; Pkg.activate(".")

# ╔═╡ 1590493c-4133-11f0-0cc3-5993612e8e01
using CairoMakie, BenchmarkTools,MLNanoShaper, MLNanoShaperRunner,GeometryBasics, Serialization, StructArrays, CUDA,FileIO, Random, Lux, DataFrames, CSV, Printf

# ╔═╡ 641d5acf-5184-4a3f-9713-31eda99317f3
using MLNanoShaperRunner: Partial, select_and_preprocess

# ╔═╡ 5b7f3312-e1db-42e1-be1c-444073305f74
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

# ╔═╡ 15633863-aeee-4233-b065-5a33278ed09d
log_paths = String[
	"tiny_soft_max_angular_dense_jobs_40_3_2025-06-04_epoch_4370_75456560920357034",
	"light_soft_max_angular_dense_jobs_40_3_2025-06-04_epoch_2870_27447953463011549",
	"light_soft_max_angular_dense_jobs_40_4_2025-05-31_epoch_560_1951733446584503143",
	"light_soft_max_angular_dense_jobs_40_5d_2025-06-04_epoch_120_8715503880527416838",
]

# ╔═╡ edf61869-8c95-423a-9832-da3a982996eb
prot_num = 2

# ╔═╡ c1e282f9-4074-4e5c-94c3-4984f7b287d7
atoms =
    getfield.(
        read("$(homedir())/datasets/pqr/$prot_num/structure.pqr", PQR{Float32}), :pos) |>
    StructVector

# ╔═╡ b7f24a37-6d26-4dd6-acef-4bfbdd60e80b
surface = load("$(homedir())/datasets/pqr/$prot_num/triangulatedSurf.off")

# ╔═╡ 42884c7b-7736-4cdd-94ce-d85da0c6b86e
points = rand(surface.position,1000)

# ╔═╡ 9363fb61-fa21-42cb-92ab-6063de3ebc73
model_weights = deserialize("$(homedir())/datasets/models/tiny_soft_max_angular_dense_jobs_40_3_2025-06-04_epoch_4370_75456560920357034")

# ╔═╡ c70ab50e-3905-414d-b38d-e538779e0858
model_weights4 = deserialize("$(homedir())/datasets/models/light_soft_max_angular_dense_jobs_40_4_2025-05-31_epoch_560_1951733446584503143")

# ╔═╡ b90eef7f-5277-4eb4-b57e-3209f5ffa2dd
function get_benchmarks(model_weights::SerializedModel,atoms::StructVector{Sphere{Float32}},points::Vector{Point3f})
	model = MLNanoShaperRunner.production_instantiate(model_weights,on_gpu=false)
	model_raw_cpu = MLNanoShaperRunner.drop_preprocessing(MLNanoShaperRunner.production_instantiate(model_weights,on_gpu=false).model)
	model_raw_gpu = MLNanoShaperRunner.drop_preprocessing(MLNanoShaperRunner.production_instantiate(model_weights,on_gpu=true).model)
	ps,st = model_weights.parameters,  model_weights.states
	cutoff_radius = get_cutoff_radius(model)
	atoms_grid = RegularGrid(atoms,cutoff_radius)
	preprocessed_data = select_and_preprocess((Batch(points),atoms_grid);cutoff_radius)
	
	bench_prepro_cpu = @benchmark select_and_preprocess($(Batch(points),atoms_grid);cutoff_radius=$cutoff_radius)
	bench_prepro_gpu = @benchmark select_and_preprocess($(Batch(points)),$atoms_grid;cutoff_radius=$cutoff_radius,device=$cu)
	bench_cpu = @benchmark Lux.apply($model_raw_cpu,$(preprocessed_data),$(ps),$(st))
	bench_gpu = @benchmark Lux.apply($model_raw_cpu,$(cu(preprocessed_data)),$(cu(ps)),$(cu(st)))
	[bench_prepro_cpu,bench_prepro_gpu,bench_cpu,bench_gpu]
end

# ╔═╡ c2279e84-dfc6-43a9-a306-b5f6b07257f1
benchmarks = get_benchmarks(model_weights,atoms,points)

# ╔═╡ 0651f85f-dbd9-470e-8e11-c69b6a04a59b
benchmarks4 = get_benchmarks(model_weights4,atoms,points)

# ╔═╡ 300ec5ae-e30c-4797-b474-45aad30c9d6a
function plot_benchmarks(vec_trials::AbstractVector{BenchmarkTools.Trial},names::AbstractVector{String})
	categories = [fill(i,length(trial.times)) for (i,trial) in enumerate(vec_trials)]
	times = [trial.times for trial in vec_trials]
	boxplot(reduce(vcat,categories),reduce(vcat,times);
			show_outliers = false,
			axis = (xticks=(eachindex(names), names),xticklabelrotation = π/4,ylabel="time(ns)",yscale = log10)
	)
end

# ╔═╡ 35abd80f-2eb0-4119-904c-6c92c0f66bf0
plot = plot_benchmarks([benchmarks[1:2]...,benchmarks4[1:2]...,benchmarks[3:end]...,benchmarks4[3:end]...],["preprocessing 3A cpu","preprocessing 3A gpu","preprocessing 4A cpu","preprocessing 4A gpu","tiny 3A cpu","tiny 3A gpu","light 4A cpu","tiny 4A gpu"])

# ╔═╡ 81fbd9f2-ed35-485b-abb2-c05dc1f0c21b
save("timeplot.pdf",plot)

# ╔═╡ 7cbe81f6-6e18-4894-9d6f-97dcbc125668
function get_dataframe_data(vec_trials::AbstractVector{BenchmarkTools.Trial},names::AbstractVector{String})
	columns = [names,["$(@sprintf("%.2f",mean(trial.times ./1e6)))($(@sprintf("%.2f",std(trial.times ./1e6))))" for trial in vec_trials]]
	DataFrame(columns,["model","time(std)(ms)"])
end

# ╔═╡ c9e6e5ca-3e57-4b46-84b2-19fb736f4263
df = get_dataframe_data([benchmarks[1:2]...,benchmarks4[1:2]...,benchmarks[3:end]...,benchmarks4[3:end]...],["preprocessing 3A cpu(ms)","preprocessing 3A gpu(ms)","preprocessing 4A cpu(ms)","preprocessing 4A gpu(ms)","tiny 3A cpu(ms)","tiny 3A gpu(ms)","light 4A cpu(ms)","tiny 4A gpu(ms)"])

# ╔═╡ f9913971-d04f-41d0-a2c0-fb8b6e01badb
CSV.write("timeplot.csv",df)

# ╔═╡ Cell order:
# ╠═b4c8f01b-5459-4a60-9a43-e85f6b2c05ee
# ╠═1590493c-4133-11f0-0cc3-5993612e8e01
# ╠═641d5acf-5184-4a3f-9713-31eda99317f3
# ╠═5b7f3312-e1db-42e1-be1c-444073305f74
# ╠═15633863-aeee-4233-b065-5a33278ed09d
# ╠═edf61869-8c95-423a-9832-da3a982996eb
# ╠═c1e282f9-4074-4e5c-94c3-4984f7b287d7
# ╠═b7f24a37-6d26-4dd6-acef-4bfbdd60e80b
# ╠═42884c7b-7736-4cdd-94ce-d85da0c6b86e
# ╠═9363fb61-fa21-42cb-92ab-6063de3ebc73
# ╠═c70ab50e-3905-414d-b38d-e538779e0858
# ╠═b90eef7f-5277-4eb4-b57e-3209f5ffa2dd
# ╠═c2279e84-dfc6-43a9-a306-b5f6b07257f1
# ╠═0651f85f-dbd9-470e-8e11-c69b6a04a59b
# ╠═300ec5ae-e30c-4797-b474-45aad30c9d6a
# ╠═35abd80f-2eb0-4119-904c-6c92c0f66bf0
# ╠═81fbd9f2-ed35-485b-abb2-c05dc1f0c21b
# ╠═7cbe81f6-6e18-4894-9d6f-97dcbc125668
# ╠═c9e6e5ca-3e57-4b46-84b2-19fb736f4263
# ╠═f9913971-d04f-41d0-a2c0-fb8b6e01badb
