### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 8cbe5092-8af4-4a2c-a574-e33cb9632058
using Pkg

# ╔═╡ 9bbec073-a5bf-429f-bc29-7411e3041888
Pkg.activate(".")

# ╔═╡ fc935a86-ceac-4d5a-8fcb-34d9c754a2f1
using MLNanoShaper, MLNanoShaperRunner, Serialization, Static, StructArrays, FileIO,
      GeometryBasics, Folds, Lux, Random,Accessors,NearestNeighbors, Statistics 

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
import CairoMakie as Mk, Meshes as Ms

# ╔═╡ ccbcea27-ea65-4b0c-8a56-c3a21fc976bb
prot_num = 2

# ╔═╡ ba125a1e-09ff-4c7f-a1d4-6da28810c0a8
dataset_dir = "$(dirname(dirname(@__FILE__)))/examples"

# ╔═╡ b91501dd-f66f-4a60-afa9-c4c9d0fc3504
names = [
"$(homedir())/datasets/models/tiny_soft_max_angular_dense_s_test35_2025-05-22_epoch_3200_7123690842925557240"
"$(homedir())/datasets/models/light_soft_max_angular_dense_s_test28_2025-05-20_epoch_750_13395197924094577505"
]

# ╔═╡ 69ee1b79-b99d-4e3a-9769-254b1939aba6
models = names.|> deserialize .|> MLNanoShaperRunner.production_instantiate

# ╔═╡ f7041ca8-97be-4998-9c10-2cbed79eb135
atoms = RegularGrid(
    getfield.(read("$dataset_dir/$prot_num/structure.pqr", PQR{Float32}), :pos) |>
    StructVector,
    3f0)

# ╔═╡ 58cf0ac8-d68d-47a7-b08f-098b65d19908
surface = load("$dataset_dir/$prot_num/triangulatedSurf.off")

# ╔═╡ a0a5f16f-0224-47b1-ae86-c4b5bd48fd07
param = [
    (; cutoff_radius = 3.0f0, default_value = 0.0f0, iso_value = 0.5f0, step = 10.0f0),
    (; cutoff_radius = 3.0f0, default_value = 0.0f0, iso_value = 0.5f0, step = 4.0f0)]

# ╔═╡ 7f3602e9-028f-44fc-b7dd-052f76438dae
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
full_data = map(MLNanoShaper.implicit_surface.(Ref(atoms), models[1:1], param[1:1])) do (points, top)
    (; points, top)
end |> StructVector
  ╠═╡ =#

# ╔═╡ ff9a463c-7742-4a6a-85b8-d9b71a7e83cd
m = models[2]

# ╔═╡ 2e208c01-0893-4ab0-a1db-51cada6a95b6
#=╠═╡
data = select_in_domaine.(((x, _, z),) -> -5 <= x <= 5, full_data)
  ╠═╡ =#

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

# ╔═╡ 4a1478c6-d200-4073-8d9d-1cbab26ff94d
invert((a, b, c)::NgonFace) = NgonFace(a, c, b)

# ╔═╡ 9f2d57c6-cc75-43ac-a431-76f92229bda4
invert((a, b, c)::Tuple) = (a, c, b)

# ╔═╡ 8b7580c3-8f2c-4d33-83bc-d368e9df19e2
# ╠═╡ disabled = true
#=╠═╡
meshes = map(data) do (; points, top)
    Ms.SimpleMesh(points .|> Tuple, top .|> Tuple .|> invert .|> Ms.connect)

end
  ╠═╡ =#

# ╔═╡ ca888076-9b41-4510-a129-2a039a8bf0ce
# ╠═╡ disabled = true
#=╠═╡
full_meshes = map(full_data) do (; points, top)
    Ms.SimpleMesh(points .|> Tuple, top .|> Tuple .|> invert .|> Ms.connect)

end
  ╠═╡ =#

# ╔═╡ cf2a31e2-502a-4be6-af8e-154b726db1ea
_ref = load("$dataset_dir/$prot_num/triangulatedSurf.off")

# ╔═╡ eec7338d-a319-478c-9cde-663e38b3e523
ref = Ms.SimpleMesh(
    coordinates(_ref) .|> Tuple, GeometryBasics.faces(_ref) .|> Tuple .|> Ms.connect)

# ╔═╡ e7e6584b-5059-46f6-a614-76866f1b1df9
#=╠═╡
begin
    f = Mk.Figure(size = (1000,700))
    Mk.Axis3(f[1, 1], title="tiny_angular_dense_cv 3A")
    Ms.viz!(f[1, 1], meshes[1]; color=:red)
    Mk.Axis3(f[1, 2], title="tiny_angular_dense_cv 2A")
    Ms.viz!(f[1, 2], meshes[2]; color=:red)
    Mk.Axis3(f[2, 1], title="tiny_angular_dense_cv 3A")loss
    Ms.viz!(f[2, 1], full_meshes[1]; color=:red)
    Ms.viz!(f[2, 1], ref; color=:green)
    a = Mk.Axis3(f[2, 2], title="tiny_angular_dense_cv 2A")
    l = Ms.viz!(f[2, 2], ref; color=:green)
    Ms.viz!(f[2, 2], full_meshes[2]; color=:red)
	Mk.Legend(f[1:2,3],[Mk.LineElement(color = :green),Mk.LineElement(color = :red)],["true value","predicted value"])
	f
end
  ╠═╡ =#

# ╔═╡ e78e5812-1927-4f67-bd3a-9bd1b577f9ad
function get_input_slice(atoms::RegularGrid, step, z)
    mins = atoms.start
	maxes = mins .+ size(atoms.grid) .* atoms.radius
    ranges = range.(mins, maxes; step)
    grid = Point3f.(reshape(ranges[1], :, 1), reshape(ranges[2], 1, :), z)
end

# ╔═╡ 2b0fc2fd-47c1-491c-b9a3-6ddff7b61850
function get_slice(atoms, model, z, (; cutoff_radius, step, default_value))
    grid = get_input_slice(atoms, step, z)
    volume = Folds.map(grid) do x
        model((MLNanoShaper.Batch([x]), atoms)) |> only
    end
end

# ╔═╡ ee6dc376-b884-4ecb-8c63-1830bd664597
slice1 = get_slice(atoms,models[1],6.0,(;cutoff_radius=3.0f0,step=.1f0,default_value=0f0))

# ╔═╡ a8a43fdf-f69c-41ef-b309-ee8531e5df23
slice2 = get_slice(atoms,models[2],6.0,(;cutoff_radius=3.0f0,step=.1f0,default_value=0.0f0))

# ╔═╡ d679ca88-615e-4675-9d0a-419cd18246f9
begin
    g = Mk.Figure(size = (1200,500))
    Mk.Axis(g[1, 1], title="tiny_angular_dense 3A")
    plt1 = Mk.plot!(g[1, 1], slice1;colormap = :rainbow)
	Mk.Colorbar(g[1, 2],plt1)
    Mk.Axis(g[1, 3], title="tiny_angular_dense 2A")
    plt2 = Mk.plot!(g[1, 3],slice2;colormap = :rainbow)
	Mk.Colorbar(g[1, 4],plt2)
	g
end

# ╔═╡ 55e62c2f-6797-4075-b8cc-d7d11e05317e
mins = atoms.start

# ╔═╡ 3c300122-c050-4ed0-9e9c-181a0698803c
maxes = mins .+ size(atoms.grid) .* atoms.radius

# ╔═╡ 67cb1851-a1a3-458f-9c9d-b5061a57ed37
ranges = range.(mins, maxes; step = 0.1)

# ╔═╡ 72e3bde9-6be8-46ac-899d-27afd99a7b3e
grid = get_input_slice(atoms, 0.1, 6)

# ╔═╡ a98327b5-f8a4-46cc-a14c-8dd234ca9933
dist = MLNanoShaper.signed_distance.(grid, Ref(MLNanoShaper.RegionMesh(surface)))

# ╔═╡ 96a915f1-08b1-4e59-bf5b-ab8f77cb39fa
Mk.plot(σ.(dist);colormap = :rainbow,colorrange = [0,1])

# ╔═╡ 44e41f3b-69c5-47f5-bb2e-b1d668eb2889
begin
	h = Mk.Figure(size = (700,500))
	Mk.Axis(h[1, 1], title="tiny_angular_dense 3A")
	Mk.contour!(h[1,1],ranges[1],ranges[2],slice1,levels=[.5],color=:red)
	Mk.contour!(h[1,1],ranges[1],ranges[2],dist,levels=[0],color = :green)
	Mk.Legend(h[1,2],[Mk.LineElement(color = :green),Mk.LineElement(color = :red)],["true value","predicted value"])
	h
end

# ╔═╡ d8dc5f29-347f-451f-897b-176c85460069
Mk.plot(map(22.65:.000001:22.75) do y m((Batch([Point3f(10,y,0)]),atoms)) |> only end)

# ╔═╡ 31803972-9bf4-462c-920d-22aa1e76f7eb
map(21.45:.00001:21.46) do y minimum(m.model.layers.layer_1.fun((Batch([Point3f(10,y,0)]),atoms)).field[5,:]) end

# ╔═╡ 42ac2eda-dfdc-4323-9185-098394477c1b
# ╠═╡ disabled = true
#=╠═╡
m = Chain(Lux.NoOpLayer(),Lux.NoOpLayer(),models[1].model[3],models[1].model[4],models[1].model[5];disable_optimizations=true)
  ╠═╡ =#

# ╔═╡ 1bec33cd-4db0-4aea-b7d1-35de8c07bbfd
ps = models[1].ps

# ╔═╡ a3f888d6-66a5-4425-9fe3-6af5d2d3f7fb
st = models[1].st

# ╔═╡ Cell order:
# ╟─5f801ac4-1f27-11ef-3246-afece906b714
# ╠═8cbe5092-8af4-4a2c-a574-e33cb9632058
# ╠═9bbec073-a5bf-429f-bc29-7411e3041888
# ╠═ea5ad13f-bf1f-4986-b781-a30a8ca83e7c
# ╠═fc935a86-ceac-4d5a-8fcb-34d9c754a2f1
# ╠═e8a48b40-28b8-41a7-a67d-cbac2b361f84
# ╠═ccbcea27-ea65-4b0c-8a56-c3a21fc976bb
# ╠═ba125a1e-09ff-4c7f-a1d4-6da28810c0a8
# ╠═b91501dd-f66f-4a60-afa9-c4c9d0fc3504
# ╠═69ee1b79-b99d-4e3a-9769-254b1939aba6
# ╠═f7041ca8-97be-4998-9c10-2cbed79eb135
# ╠═58cf0ac8-d68d-47a7-b08f-098b65d19908
# ╠═a0a5f16f-0224-47b1-ae86-c4b5bd48fd07
# ╠═7f3602e9-028f-44fc-b7dd-052f76438dae
# ╠═ff9a463c-7742-4a6a-85b8-d9b71a7e83cd
# ╠═2e208c01-0893-4ab0-a1db-51cada6a95b6
# ╠═d38242b4-0bee-44c8-9885-42e8441faf25
# ╠═4a1478c6-d200-4073-8d9d-1cbab26ff94d
# ╠═9f2d57c6-cc75-43ac-a431-76f92229bda4
# ╠═8b7580c3-8f2c-4d33-83bc-d368e9df19e2
# ╠═ca888076-9b41-4510-a129-2a039a8bf0ce
# ╠═cf2a31e2-502a-4be6-af8e-154b726db1ea
# ╠═eec7338d-a319-478c-9cde-663e38b3e523
# ╠═e7e6584b-5059-46f6-a614-76866f1b1df9
# ╠═e78e5812-1927-4f67-bd3a-9bd1b577f9ad
# ╠═2b0fc2fd-47c1-491c-b9a3-6ddff7b61850
# ╠═ee6dc376-b884-4ecb-8c63-1830bd664597
# ╠═a8a43fdf-f69c-41ef-b309-ee8531e5df23
# ╠═d679ca88-615e-4675-9d0a-419cd18246f9
# ╠═55e62c2f-6797-4075-b8cc-d7d11e05317e
# ╠═3c300122-c050-4ed0-9e9c-181a0698803c
# ╠═67cb1851-a1a3-458f-9c9d-b5061a57ed37
# ╠═72e3bde9-6be8-46ac-899d-27afd99a7b3e
# ╠═a98327b5-f8a4-46cc-a14c-8dd234ca9933
# ╠═96a915f1-08b1-4e59-bf5b-ab8f77cb39fa
# ╠═44e41f3b-69c5-47f5-bb2e-b1d668eb2889
# ╠═d8dc5f29-347f-451f-897b-176c85460069
# ╠═31803972-9bf4-462c-920d-22aa1e76f7eb
# ╠═42ac2eda-dfdc-4323-9185-098394477c1b
# ╠═1bec33cd-4db0-4aea-b7d1-35de8c07bbfd
# ╠═a3f888d6-66a5-4425-9fe3-6af5d2d3f7fb
