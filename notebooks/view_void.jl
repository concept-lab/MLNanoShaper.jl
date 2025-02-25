### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 7d8db5ba-3eaa-11ef-241f-c5fafd5576eb
using Pkg

# ╔═╡ 053fdeb3-7df9-42dd-88e4-8e4513ae5933
Pkg.activate(".")

# ╔═╡ 0a309e3a-7e97-4a7a-ae6b-8d6aab584907
using MLNanoShaper, MLNanoShaperRunner, Serialization, Static, StructArrays, FileIO,
      GeometryBasics, Folds, Lux, CUDA

# ╔═╡ e84880c5-9857-453e-88af-b9f49d084dae
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

# ╔═╡ aeb86d5d-7fe9-400f-9181-2231d2f0d6ff
import CairoMakie as Mk, Meshes as Ms

# ╔═╡ 65b41633-4f9e-4269-b4f0-eaee047c205b
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 08e5ce55-f18f-459e-a6b4-0f06d3e14c58
atoms = MLNanoShaper.AnnotedKDTree(
    read("fullerene.xyzr", MLNanoShaperRunner.Import.XYZR{Float32}) |> StructVector,
    static(:center))

# ╔═╡ f4e5947b-d676-42b9-a690-35af64178484
param = [
    (; cutoff_radius = 3.0f0, default_value = -10.0f0, iso_value = 0.0f0, step = 10.0f0),
    (; cutoff_radius = 2.0f0, default_value = 0.0f0, iso_value = 0.5f0, step = 4.0f0)]

# ╔═╡ 24225374-d1a1-4b64-9bb3-3145cfab23ed
names = ["$(homedir())/datasets/models/tiny_angular_dense_s_jobs_1_8_2_c_2025-02-25_epoch_90_9904683113990176820"
"$(homedir())/datasets/models/tiny_angular_dense_s_jobs_1_7_3_c_2025-02-25_epoch_90_5467308422902619215"]

# ╔═╡ 9bb279e8-4bee-42b9-806f-b5d024df5fca
models = names .|> deserialize .|> MLNanoShaper.extract_model .|> gpu_device()

# ╔═╡ fc5e5ddb-29a5-438e-94c3-b92130709d30
function get_input_slice(atoms, step, z)
    (; mins, maxes) = atoms.tree.hyper_rec
    ranges = range.(mins .- 2, maxes .+ 2; step)
    grid = Point3f.(reshape(ranges[1], :, 1), reshape(ranges[2], 1, :), z)
end

# ╔═╡ 265fdba5-de9b-4831-98eb-1eef51baa7bf
function get_slice(atoms, model, z, (; cutoff_radius, step, default_value))
    grid = get_input_slice(atoms, step, z)
    volume = Folds.map(grid) do x
        evaluate_model(model, x, atoms; cutoff_radius, default_value)
    end
end

# ╔═╡ 7582574c-a307-43d8-a1bd-e07774c7deb4
step = 0.01f0

# ╔═╡ d3091b31-cc8d-4668-acd4-c2e8db0fc723
slice1 = get_slice(
    atoms, models[1], 0.0, (; cutoff_radius = 3.0f0, step, default_value = -10.0f0))

# ╔═╡ db5ea403-dabc-4825-a293-4f43945ca0f2
slice2 = get_slice(
    atoms, models[2], 0.0, (; cutoff_radius = 2.0f0, step, default_value = 0.0f0))

# ╔═╡ 13633be8-3cb0-4a52-86d8-0ea9c0ad82cb
(; mins, maxes) = atoms.tree.hyper_rec

# ╔═╡ 68758557-934a-4c8b-99b0-135d7c6b2b9c
ranges = range.(mins .- 2, maxes .+ 2; step)

# ╔═╡ dc532ad4-3f40-464e-a1ee-304aa2fe36ef
begin
    h = Mk.Figure(size = (500, 500))
    Mk.Axis(h[1, 1], title = "tiny_angular_dense 3A")
    Mk.contour!(h[1, 1], ranges[1], ranges[2], slice1, levels = [0.5], color = :red)
    h
end

# ╔═╡ 3649ef90-e227-49f2-bd3d-ad93aa62a465
function alloc_concatenated(sub_array, l)
    similar(
        sub_array,
        sub_array |> eltype,
        (size(sub_array)[begin:(end - 1)]..., l))
end

# ╔═╡ 350ea3d8-7d1c-4902-89b2-efc83c48ffdc
function _kernel_sum!(a::CuDeviceMatrix{T}, b::CuDeviceMatrix{T},
        nb_elements::CuDeviceVector{Int}) where {T}
    nb_lines = size(b, 1)
    identifiant = threadIdx().x + blockDim().x * blockIdx().x
    i, n = identifiant % nb_lines + 1, identifiant ÷ nb_lines + 1
    if n + 1 > length(nb_elements)
        # we are launching more threads than required
        return
    end
    a[i, n] = zero(T)
    for j in (nb_elements[n] + 1):nb_elements[n + 1]
        a[i, n] += b[i, j]
    end
end

# ╔═╡ 28dfc4bf-2866-421b-abb7-a70c9325dae2
function batched_sum!(a::CuMatrix, b::CuMatrix, nb_elements::CuVector{Int})
    nb_computations = size(b, 1) * (length(nb_elements) - 1)
    block_size = 1024
    @cuda threads=block_sizeblocks = (1 + (nb_computations - 1) ÷ block_size) _kernel_sum!(
        a, b, nb_elements)
end

# ╔═╡ 5832320e-a113-4c18-938c-e26886419970
function batched_sum(b::CuMatrix, nb_elements::Vector{Int})
    a = similar(b, eltype(b), (size(b, 1), length(nb_elements)))
    batched_sum!(a, b, cu(nb_elements))
    a
end

# ╔═╡ 56f97e4b-5b1a-494f-86b6-ff4f4f5aba48
function evaluate_and_cat(arrays, n::Int, sub_array, get_slice)
    indexes = 1:n
    res = alloc_concatenated(sub_array, get_slice(n) |> last)
    foreach(indexes) do i
        view(res, fill(:, ndims(sub_array) - 1)..., get_slice(i)) .= arrays(i)
    end
    res
end

# ╔═╡ fec6ad0a-09b7-4862-8ec6-d7b27da07260
begin
    g = Mk.Figure(size = (1200, 500))
    Mk.Axis(g[1, 1], title = "tiny_angular_dense 3A")
    plt1 = Mk.plot!(g[1, 1], exp.(slice1) ./ (exp.(-slice1) .+ exp.(slice1));
        colormap = :rainbow, colorrange = [0, 1])
    Mk.Colorbar(g[1, 2], plt1)
    Mk.Axis(g[1, 3], title = "tiny_angular_dense 2A")
    plt2 = Mk.plot!(g[1, 3], slice2; colormap = :rainbow, colorrange = [0, 1])
    Mk.Colorbar(g[1, 4], plt2)
    g
end

# ╔═╡ Cell order:
# ╠═7d8db5ba-3eaa-11ef-241f-c5fafd5576eb
# ╠═053fdeb3-7df9-42dd-88e4-8e4513ae5933
# ╠═e84880c5-9857-453e-88af-b9f49d084dae
# ╠═0a309e3a-7e97-4a7a-ae6b-8d6aab584907
# ╠═aeb86d5d-7fe9-400f-9181-2231d2f0d6ff
# ╠═65b41633-4f9e-4269-b4f0-eaee047c205b
# ╠═08e5ce55-f18f-459e-a6b4-0f06d3e14c58
# ╠═f4e5947b-d676-42b9-a690-35af64178484
# ╠═24225374-d1a1-4b64-9bb3-3145cfab23ed
# ╠═9bb279e8-4bee-42b9-806f-b5d024df5fca
# ╠═fc5e5ddb-29a5-438e-94c3-b92130709d30
# ╠═265fdba5-de9b-4831-98eb-1eef51baa7bf
# ╠═7582574c-a307-43d8-a1bd-e07774c7deb4
# ╠═d3091b31-cc8d-4668-acd4-c2e8db0fc723
# ╠═db5ea403-dabc-4825-a293-4f43945ca0f2
# ╠═13633be8-3cb0-4a52-86d8-0ea9c0ad82cb
# ╠═68758557-934a-4c8b-99b0-135d7c6b2b9c
# ╠═dc532ad4-3f40-464e-a1ee-304aa2fe36ef
# ╠═3649ef90-e227-49f2-bd3d-ad93aa62a465
# ╠═350ea3d8-7d1c-4902-89b2-efc83c48ffdc
# ╠═28dfc4bf-2866-421b-abb7-a70c9325dae2
# ╠═5832320e-a113-4c18-938c-e26886419970
# ╠═56f97e4b-5b1a-494f-86b6-ff4f4f5aba48
# ╠═fec6ad0a-09b7-4862-8ec6-d7b27da07260
