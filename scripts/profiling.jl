### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 28210014-39eb-11ef-24eb-110acb81da08
using Pkg,Revise

# ╔═╡ de53cb27-dbef-4e3d-9d12-0f3d1b5acbc4
Pkg.activate(".")

# ╔═╡ e9f0f433-0fe9-4096-b484-b432ec54afc8
using MLNanoShaper, MLNanoShaperRunner, FileIO, StructArrays, Static, Serialization,GeometryBasics, LuxCUDA, Lux, Profile, ProfileSVG, ChainRulesCore,Folds,BenchmarkTools, Zygote,Distances, LinearAlgebra

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
import WGLMakie as Mk

# ╔═╡ 26cb955f-13ac-43cf-90d1-516b729ecf3a
function alloc_concatenated(sub_array,l)
	similar(
            sub_array,
            sub_array |> eltype,
			(size(sub_array)[begin:end-1]...,  l))
end

# ╔═╡ 35484485-c8c5-41bd-8d06-61c92f6cae7a
A = [1; 2 ; 3;;
	 4; 5 ; 6;;;
	 7; 8 ; 9;;
	 10;11;12]

# ╔═╡ 86eb31a4-9312-4a28-abae-d76227c7bf87
function _kernel_sum!(a::CuDeviceMatrix{T},b::CuDeviceMatrix{T},nb_elements::CuDeviceVector{Int}) where T
	nb_lines = size(b,1)
	identifiant = threadIdx().x + blockDim().x * blockIdx().x
	i,n = identifiant% nb_lines + 1, identifiant ÷ nb_lines + 1
	if n +1 > length(nb_elements)
		# we are launching mor threads than required
		return
	end
	a[i,n] = zero(T)
	for j in (nb_elements[n]+1):nb_elements[n+1]
		a[i,n] += b[i,j]
	end
end

# ╔═╡ b9986ae1-4de9-44f9-9702-2ff11c683926
function batched_sum!(a::CuMatrix,b::CuMatrix,nb_elements::CuVector{Int})
	nb_computations = size(b,1)*(length(nb_elements)-1) 
	block_size=1024
	@cuda threads=block_sizeblocks=(1+(nb_computations -1) ÷ block_size) _kernel_sum!(a,b,nb_elements)	
end

# ╔═╡ e4502591-8b74-4aa7-837d-c202bede0b8e
function batched_sum(b::CuMatrix,nb_elements::Vector{Int})
	a = similar(b,eltype(b),(size(b,1),length(nb_elements)))
	batched_sum!(a,b,cu(nb_elements))
	a
end

# ╔═╡ 48bd4a94-48f3-4f40-869e-dee0bf0c52ee
function evaluate_and_cat(arrays,n::Int,sub_array,get_slice)
	indexes=1:n
	res = alloc_concatenated(sub_array , get_slice(n) |> last)
	foreach(indexes) do i 
		view(res,fill(:, ndims(sub_array) - 1)..., get_slice(i)) .= arrays(i)
	end
	res
end

# ╔═╡ f560fd11-65a0-4252-a8e0-c1bcf0e2fa3b
# ╠═╡ disabled = true
#=╠═╡
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(evaluate_and_cat), arrays, n::Int,sub_array,get_slice)
	indexes=1:n
	res = alloc_concatenated(sub_array , get_slice(n) |> last)
	pullbacks = Array{Function}(undef,n)
	Folds.foreach(indexes) do i 
		res[fill(:, ndims(sub_array) - 1)..., get_slice(i)],pullbacks[i] =  rrule_via_ad(config,arrays,i)
	end
	function pullback_evaluate_and_cat(dres)
		map(indexes) do i
			pullbacks[i](dres[fill(:, ndims(sub_array) - 1)..., get_slice(i)])
		end,NoTangent(),NoTangent(),NoTangent(),NoTangent()
	end
		res,pullback_evaluate_and_cat
end
  ╠═╡ =#

# ╔═╡ 5b4b8b8f-f712-4c7c-8182-2d5a7d07ea48
function make_id_product(n)
	mapreduce(vcat,1:n) do i
		1:i
	end,mapreduce(vcat,1:n) do i
		fill(i,1:i)
	end
end

# ╔═╡ 8036d772-af13-4c26-8d83-db295b5e2340
function MLNanoShaperRunner.preprocessing((; point, atoms)::ModelInput{T}) where {T}
    prod = make_id_product(length(atoms))
	n_tot = length(prod[1])
	res = StructArray{PreprocessData{T}}(undef,n_tot)
	for n in 1:n_tot
        res.d_1[n] = euclidean(point, atoms[prod[1][n]].center)
	end
	for n in 1:n_tot
		res.r_1[n] = atoms[prod[1][n]].r
	end
	for n in 1:n_tot
		res.d_2[n] = euclidean(point, atoms[prod[2][n]].center)
	end
	for n in 1:n_tot
		res.r_2[n] = atoms[prod[2][n]].r
	end
	for n in 1:n_tot
		res.dot[n] = (atoms[prod[1][n]].center - point) ⋅ (atoms[prod[2][n]].center - point) / (res.d_1[n]  * res.d_2[n] + 1.0f-8)
	end
	reshape(res,1,:)
end

# ╔═╡ a009ddb6-ffc7-42ed-8964-d0e763adb312
MLNanoShaperRunner.tiny_angular_dense()

# ╔═╡ 5765cbc5-ec12-406e-b43f-9291c99b9d1d
prot_num=1

# ╔═╡ 2cce8a1a-97fe-45ae-bca7-584b843739d6
surface= load("$(homedir())/datasets/pqr/$prot_num/triangulatedSurf.off")

# ╔═╡ d9898910-1b29-400c-bcea-457017723c70
atoms = MLNanoShaperRunner.AnnotedKDTree(getfield.(read("$(homedir())/datasets/pqr/$prot_num/structure.pqr", PQR{Float32}), :pos) |> StructVector, static(:center))

# ╔═╡ 0adf29e7-a6e4-48ae-bfe0-e5340d1d1a70
model = "$(homedir())/datasets/models/tiny_angular_dense_c_2.0A_continuous_1_2024-07-04_epoch_40_9139422370875205944"|>
deserialize|> 
MLNanoShaper.extract_model|> 
gpu_device()

# ╔═╡ 93dde6e9-cd8d-4147-9106-c5f9c78115b4
"$(homedir())/datasets/models/tiny_angular_dense_c_2.0A_continuous_1_2024-07-04_epoch_40_9139422370875205944" |> deserialize

# ╔═╡ 0d81f6cb-c6d4-4748-b23c-46ab1b0f9a8e
cutoff_radius=3.0f0

# ╔═╡ 2ff3238a-2205-405a-9075-553f27db84d6
default_value=-8f0

# ╔═╡ 3ed5f526-99fd-4e91-b7f0-6bc7d8c67afa
atoms.tree.hyper_rec

# ╔═╡ f44f276c-7de9-49dc-b584-5a64a04006a0
51*56*48/4000* .03

# ╔═╡ 045ef25b-c9ea-42a9-ab00-c8f2b7994bde
x=atoms.data.center

# ╔═╡ f2343acb-23c2-4155-b393-c0bcea4d9760
@benchmark  model((MLNanoShaperRunner.Batch(x), atoms) )

# ╔═╡ c613a4d0-59de-4726-81cc-6a073602cbf9
CUDA.@profile  model.model((MLNanoShaperRunner.Batch(x), atoms),model.ps,model.st)  |> first


# ╔═╡ 26bc9d17-35ca-4900-a99b-6156ee251920
methods(Base.mapreducedim!)

# ╔═╡ 5273eaf5-2762-41ba-b634-17e170adc65e
begin
	Profile.clear()
	Profile.init(n = 10^7, delay = 0.00000001)
	@profile  model.model((MLNanoShaperRunner.Batch(x), atoms),model.ps,model.st)  |> first
	ProfileSVG.view(maxdepth=100)
end

# ╔═╡ a20995b8-337f-44e0-ba7a-aeb17f8d3eb7
y = model.model.layers.layer_1((MLNanoShaperRunner.Batch(x), atoms)).field

# ╔═╡ 02cf980b-e542-41a4-aa5c-a5b62d1192a1
lengths = vcat([0], y .|> size .|> last |> cumsum)

# ╔═╡ 857b6847-3556-4435-909a-d3e5cbb0b22a
begin
	batched = similar(
		y |> first,
		y |> first |> eltype,
		(size(y |> first)[begin:end-1]...,  last(lengths)))
	Folds.foreach(lengths[2:end-1] |> eachindex) do i
		batched[fill(:, ndims(y |> first) - 1)...,  (lengths[i]+1:lengths[i+1])] = y[i]
	end
	batched
end

# ╔═╡ 4e349f8d-327b-416f-b938-b8db9b9b6f87
vec(batched)

# ╔═╡ 98639c39-2953-4b67-bdb6-7c2ebab19c90
batched[fill(:, ndims(y |> first) - 1)..., (lengths[1]+1:lengths[2])]

# ╔═╡ bc825b42-eaf9-478d-a29c-61d5e2745262
z = MLNanoShaperRunner.symetrise(y;cutoff_radius=2f0)

# ╔═╡ b8dd9e29-4cce-457b-808e-ff206f4e39da
z1 =model.model.layers.layer_2.prepross.layers.layer_2.func(z)

# ╔═╡ 4bc276f3-88f9-4a2d-b5ee-61395d09a1f8
model.model.layers.layer_2.prepross.layers.layer_3.layers.layer_1.layers.layer_1(
	z1,
	model.ps.layer_2.layer_3.layer_1.layer_1,
	model.st.layer_2.layer_3.layer_1.layer_1
)

# ╔═╡ 2d050624-7598-4068-904d-17a603864dd1
model.ps.layer_2.layer_3.layer_1.layer_1

# ╔═╡ be10222a-f6c1-4ebd-93b8-90f382003d3b
model.st.layer_2.layer_3.layer_1

# ╔═╡ 2f53d050-86c8-499b-8f99-be7be34f5632
function MLNanoShaperRunner.trace(message::String,x)
	x
end

# ╔═╡ b1294b09-c1fd-46dc-b905-fb844593a444
function (f::MLNanoShaperRunner.DeepSet)(arg::Batch, ps, st::NamedTuple)
    lengths = vcat([0], arg.field .|> size .|> last |> cumsum)
	get_slice(i) =  (lengths[i]+1:lengths[i+1])
    batched = evaluate_and_cat(length(arg.field),arg.field|> first,get_slice) do i
		arg.field[i]
	end
    res::AbstractMatrix{<:Number} = Lux.apply(f.prepross, batched, ps, st) |> first |> trace("raw")
    @assert size(res, 2) == last(lengths)
	batched_sum(res,lengths), st
end

# ╔═╡ 42da753c-7368-4ddb-9894-5862fe2a017d
trace("atoms",1)

# ╔═╡ Cell order:
# ╠═e4fc0299-2b72-4b8f-940d-9f55a76f83ca
# ╠═28210014-39eb-11ef-24eb-110acb81da08
# ╠═de53cb27-dbef-4e3d-9d12-0f3d1b5acbc4
# ╠═1c0a8115-da6e-4b09-a9ac-17672c8b73d2
# ╠═e9f0f433-0fe9-4096-b484-b432ec54afc8
# ╠═26cb955f-13ac-43cf-90d1-516b729ecf3a
# ╠═35484485-c8c5-41bd-8d06-61c92f6cae7a
# ╠═86eb31a4-9312-4a28-abae-d76227c7bf87
# ╠═b9986ae1-4de9-44f9-9702-2ff11c683926
# ╠═e4502591-8b74-4aa7-837d-c202bede0b8e
# ╠═48bd4a94-48f3-4f40-869e-dee0bf0c52ee
# ╠═f560fd11-65a0-4252-a8e0-c1bcf0e2fa3b
# ╠═b1294b09-c1fd-46dc-b905-fb844593a444
# ╠═5b4b8b8f-f712-4c7c-8182-2d5a7d07ea48
# ╠═8036d772-af13-4c26-8d83-db295b5e2340
# ╠═a009ddb6-ffc7-42ed-8964-d0e763adb312
# ╠═5765cbc5-ec12-406e-b43f-9291c99b9d1d
# ╠═2cce8a1a-97fe-45ae-bca7-584b843739d6
# ╠═d9898910-1b29-400c-bcea-457017723c70
# ╠═0adf29e7-a6e4-48ae-bfe0-e5340d1d1a70
# ╠═93dde6e9-cd8d-4147-9106-c5f9c78115b4
# ╠═0d81f6cb-c6d4-4748-b23c-46ab1b0f9a8e
# ╠═2ff3238a-2205-405a-9075-553f27db84d6
# ╠═3ed5f526-99fd-4e91-b7f0-6bc7d8c67afa
# ╠═f44f276c-7de9-49dc-b584-5a64a04006a0
# ╠═045ef25b-c9ea-42a9-ab00-c8f2b7994bde
# ╠═f2343acb-23c2-4155-b393-c0bcea4d9760
# ╠═c613a4d0-59de-4726-81cc-6a073602cbf9
# ╠═26bc9d17-35ca-4900-a99b-6156ee251920
# ╠═5273eaf5-2762-41ba-b634-17e170adc65e
# ╠═02cf980b-e542-41a4-aa5c-a5b62d1192a1
# ╠═857b6847-3556-4435-909a-d3e5cbb0b22a
# ╠═4e349f8d-327b-416f-b938-b8db9b9b6f87
# ╠═98639c39-2953-4b67-bdb6-7c2ebab19c90
# ╠═a20995b8-337f-44e0-ba7a-aeb17f8d3eb7
# ╠═bc825b42-eaf9-478d-a29c-61d5e2745262
# ╠═b8dd9e29-4cce-457b-808e-ff206f4e39da
# ╠═4bc276f3-88f9-4a2d-b5ee-61395d09a1f8
# ╠═2d050624-7598-4068-904d-17a603864dd1
# ╠═be10222a-f6c1-4ebd-93b8-90f382003d3b
# ╠═2f53d050-86c8-499b-8f99-be7be34f5632
# ╠═42da753c-7368-4ddb-9894-5862fe2a017d
