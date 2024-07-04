### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 28210014-39eb-11ef-24eb-110acb81da08
using Pkg,Revise

# ╔═╡ de53cb27-dbef-4e3d-9d12-0f3d1b5acbc4
Pkg.activate(".")

# ╔═╡ e9f0f433-0fe9-4096-b484-b432ec54afc8
using MLNanoShaper, MLNanoShaperRunner, FileIO, StructArrays, Static, Serialization,GeometryBasics

# ╔═╡ 1c0a8115-da6e-4b09-a9ac-17672c8b73d2
import WGLMakie as Mk

# ╔═╡ a009ddb6-ffc7-42ed-8964-d0e763adb312
MLNanoShaperRunner.tiny_angular_dense()

# ╔═╡ 5765cbc5-ec12-406e-b43f-9291c99b9d1d
prot_num=1

# ╔═╡ 2cce8a1a-97fe-45ae-bca7-584b843739d6
surface= load("$(homedir())/datasets/pqr/$prot_num/triangulatedSurf.off")

# ╔═╡ d9898910-1b29-400c-bcea-457017723c70
atoms = MLNanoShaperRunner.AnnotedKDTree(getfield.(read("$(homedir())/datasets/pqr/$prot_num/structure.pqr", PQR{Float32}), :pos) |> StructVector, static(:center))

# ╔═╡ 0adf29e7-a6e4-48ae-bfe0-e5340d1d1a70
model = "$(homedir())/datasets/models/tiny_angular_dense_c_2.0A_continuous_1_2024-07-04_epoch_40_9139422370875205944" |> deserialize |> MLNanoShaper.extract_model

# ╔═╡ 93dde6e9-cd8d-4147-9106-c5f9c78115b4
"$(homedir())/datasets/models/tiny_angular_dense_c_2.0A_continuous_1_2024-07-04_epoch_40_9139422370875205944" |> deserialize

# ╔═╡ 0d81f6cb-c6d4-4748-b23c-46ab1b0f9a8e
cutoff_radius=3.0f0

# ╔═╡ 2ff3238a-2205-405a-9075-553f27db84d6
default_value=-8f0

# ╔═╡ 045ef25b-c9ea-42a9-ab00-c8f2b7994bde
x=Point3f(1,0,0)

# ╔═╡ f48ec31b-2e32-4002-bfed-a32099516a2e
ENV["JULIA_DEBUG"] = "all"

# ╔═╡ b9f8d8be-af33-412c-92e2-f5adbbfbe224
@debug "aaa"

# ╔═╡ c613a4d0-59de-4726-81cc-6a073602cbf9
model((MLNanoShaperRunner.Batch[x], atoms))

# ╔═╡ 2f53d050-86c8-499b-8f99-be7be34f5632
function MLNanoShaperRunner.trace(message::String,x)
	@info message,x
	x
end

# ╔═╡ 42da753c-7368-4ddb-9894-5862fe2a017d
trace("atoms",1)

# ╔═╡ 8e246e2e-74bf-4a9c-9d9a-7c292e5aa3a1
y = [0 0; 1 1]

# ╔═╡ d14572a4-6e0f-4f45-8019-76bbe1dfb9af
y[end:end,:]

# ╔═╡ Cell order:
# ╠═28210014-39eb-11ef-24eb-110acb81da08
# ╠═de53cb27-dbef-4e3d-9d12-0f3d1b5acbc4
# ╠═1c0a8115-da6e-4b09-a9ac-17672c8b73d2
# ╠═e9f0f433-0fe9-4096-b484-b432ec54afc8
# ╠═a009ddb6-ffc7-42ed-8964-d0e763adb312
# ╠═5765cbc5-ec12-406e-b43f-9291c99b9d1d
# ╠═2cce8a1a-97fe-45ae-bca7-584b843739d6
# ╠═d9898910-1b29-400c-bcea-457017723c70
# ╠═0adf29e7-a6e4-48ae-bfe0-e5340d1d1a70
# ╠═93dde6e9-cd8d-4147-9106-c5f9c78115b4
# ╠═0d81f6cb-c6d4-4748-b23c-46ab1b0f9a8e
# ╠═2ff3238a-2205-405a-9075-553f27db84d6
# ╠═045ef25b-c9ea-42a9-ab00-c8f2b7994bde
# ╠═f48ec31b-2e32-4002-bfed-a32099516a2e
# ╠═b9f8d8be-af33-412c-92e2-f5adbbfbe224
# ╠═c613a4d0-59de-4726-81cc-6a073602cbf9
# ╠═2f53d050-86c8-499b-8f99-be7be34f5632
# ╠═42da753c-7368-4ddb-9894-5862fe2a017d
# ╠═8e246e2e-74bf-4a9c-9d9a-7c292e5aa3a1
# ╠═d14572a4-6e0f-4f45-8019-76bbe1dfb9af
