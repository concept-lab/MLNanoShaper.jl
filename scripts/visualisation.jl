### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 8cbe5092-8af4-4a2c-a574-e33cb9632058
using Pkg

# ╔═╡ 9bbec073-a5bf-429f-bc29-7411e3041888
Pkg.activate(".")

# ╔═╡ fc935a86-ceac-4d5a-8fcb-34d9c754a2f1
using MLNanoShaper, MLNanoShaperRunner, Serialization, Static, StructArrays, FileIO, GeometryBasics

# ╔═╡ 5f801ac4-1f27-11ef-3246-afece906b714
md"""
This script is used to generate the various figures in the final report.
To run this script, please run `using Pluto;Pluto.run` in a new Julia shell.

"""

# ╔═╡ ea5ad13f-bf1f-4986-b781-a30a8ca83e7c
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

# ╔═╡ e8a48b40-28b8-41a7-a67d-cbac2b361f84
import GLMakie as Mk, Meshes as Ms

# ╔═╡ 69ee1b79-b99d-4e3a-9769-254b1939aba6
models = ["$(homedir())/datasets/models/angular_dense_3Aef3_2024-06-03_epoch_100_12910378923337060099"
"$(homedir())/datasets/models/deep_angular_dense_3Aef3_2024-06-03_epoch_100_12855753833200435595"] .|> deserialize .|> MLNanoShaper.extract_model


# ╔═╡ f7041ca8-97be-4998-9c10-2cbed79eb135
atoms = MLNanoShaperRunner.AnnotedKDTree(getfield.(read("/home/tristan/datasets/pqr/1/structure.pqr", PQR{Float32}), :pos) |> StructVector, static(:center))

# ╔═╡ 7f3602e9-028f-44fc-b7dd-052f76438dae
full_data = map(MLNanoShaper.implicit_surface.(Ref(atoms), models, Ref((; cutoff_radius=3.0f0, step=1.0f0)))) do (points, top)
    (; points, top)
end |> StructVector

# ╔═╡ d38242b4-0bee-44c8-9885-42e8441faf25
function select_in_domaine(predicate, (; points, top))
    top = filter(top) do top
        any(top) do id
            points[id] |> predicate
        end
    end
    #point = filter(predicate,point)
    (; points, top)
end

# ╔═╡ 2e208c01-0893-4ab0-a1db-51cada6a95b6
data = select_in_domaine.(((x, _, z),) -> -10 <= x <= 10, full_data)

# ╔═╡ 4a1478c6-d200-4073-8d9d-1cbab26ff94d
invert((a, b, c)::NgonFace) = NgonFace(a, c, b)

# ╔═╡ 9f2d57c6-cc75-43ac-a431-76f92229bda4
invert((a, b, c)::Tuple) = (a, c, b)

# ╔═╡ 8b7580c3-8f2c-4d33-83bc-d368e9df19e2
meshes = map(data) do (; points, top)
    Ms.SimpleMesh(points .|> Tuple, top .|> Tuple .|> invert .|> Ms.connect)

end

# ╔═╡ ca888076-9b41-4510-a129-2a039a8bf0ce
full_meshes = map(full_data) do (; points, top)
    Ms.SimpleMesh(points .|> Tuple, top .|> Tuple .|> invert .|> Ms.connect)

end

# ╔═╡ cf2a31e2-502a-4be6-af8e-154b726db1ea
_ref = load("$(homedir())/datasets/pqr/1/triangulatedSurf.off")

# ╔═╡ eec7338d-a319-478c-9cde-663e38b3e523
ref = Ms.SimpleMesh(coordinates(_ref) .|> Tuple, GeometryBasics.faces(_ref) .|> Tuple .|> Ms.connect)

# ╔═╡ e7e6584b-5059-46f6-a614-76866f1b1df9
begin
    f = Mk.Figure(size = (1000,700))
    Mk.Axis3(f[1, 1], title="angular_dense")
    Ms.viz!(f[1, 1], meshes[1]; color=:red)
    Mk.Axis3(f[1, 2], title="deep_angular_dense")
    Ms.viz!(f[1, 2], meshes[2]; color=:red)
    Mk.Axis3(f[2, 1], title="angular_dense")
    Ms.viz!(f[2, 1], full_meshes[1]; color=:red)
    Ms.viz!(f[2, 1], ref; color=:green)
    a = Mk.Axis3(f[2, 2], title="deep_angular_dense")
    l = Ms.viz!(f[2, 2], ref; color=:green)
    Ms.viz!(f[2, 2], full_meshes[2]; color=:red)
	Mk.Legend(f[1:2,3],[Mk.LineElement(color = :green),Mk.LineElement(color = :red)],["true value","predicted value"])
	f
end

# ╔═╡ Cell order:
# ╟─5f801ac4-1f27-11ef-3246-afece906b714
# ╠═8cbe5092-8af4-4a2c-a574-e33cb9632058
# ╠═9bbec073-a5bf-429f-bc29-7411e3041888
# ╟─ea5ad13f-bf1f-4986-b781-a30a8ca83e7c
# ╠═fc935a86-ceac-4d5a-8fcb-34d9c754a2f1
# ╠═e8a48b40-28b8-41a7-a67d-cbac2b361f84
# ╠═69ee1b79-b99d-4e3a-9769-254b1939aba6
# ╠═f7041ca8-97be-4998-9c10-2cbed79eb135
# ╠═7f3602e9-028f-44fc-b7dd-052f76438dae
# ╠═2e208c01-0893-4ab0-a1db-51cada6a95b6
# ╠═d38242b4-0bee-44c8-9885-42e8441faf25
# ╠═4a1478c6-d200-4073-8d9d-1cbab26ff94d
# ╠═9f2d57c6-cc75-43ac-a431-76f92229bda4
# ╠═8b7580c3-8f2c-4d33-83bc-d368e9df19e2
# ╠═ca888076-9b41-4510-a129-2a039a8bf0ce
# ╠═cf2a31e2-502a-4be6-af8e-154b726db1ea
# ╠═eec7338d-a319-478c-9cde-663e38b3e523
# ╠═e7e6584b-5059-46f6-a614-76866f1b1df9