### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 8cbe5092-8af4-4a2c-a574-e33cb9632058
using Pkg

# ╔═╡ 9bbec073-a5bf-429f-bc29-7411e3041888
Pkg.activate(".")

# ╔═╡ fc935a86-ceac-4d5a-8fcb-34d9c754a2f1
using Revise,MLNanoShaper, MLNanoShaperRunner, Serialization, Static, StructArrays, FileIO, GeometryBasics, Folds, Lux

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
import WGLMakie as Mk, Meshes as Ms

# ╔═╡ ccbcea27-ea65-4b0c-8a56-c3a21fc976bb
prot_num = 1

# ╔═╡ 69ee1b79-b99d-4e3a-9769-254b1939aba6
models = [
"$(homedir())/datasets/models/tiny_angular_dense_cv_3.0A_small_grid_3_2024-06-28_epoch_100_14010870659515058109"
"$(homedir())/datasets/models/tiny_angular_dense_v_2.0A_small_grid_4_2024-07-02_epoch_50_9474739815235648407"
] .|> deserialize .|> MLNanoShaper.extract_model .|> gpu_device()


# ╔═╡ f7041ca8-97be-4998-9c10-2cbed79eb135
atoms = MLNanoShaperRunner.AnnotedKDTree(getfield.(read("$(homedir())/datasets/pqr/$prot_num/structure.pqr", PQR{Float32}), :pos) |> StructVector, static(:center))

# ╔═╡ 58cf0ac8-d68d-47a7-b08f-098b65d19908
surface= load("$(homedir())/datasets/pqr/$prot_num/triangulatedSurf.off")

# ╔═╡ a0a5f16f-0224-47b1-ae86-c4b5bd48fd07
param = [(; cutoff_radius=3.0f0, default_value = -10f0,iso_value=0f0,step=10.0f0),(; cutoff_radius=2.0f0, default_value = 0f0,iso_value=.5f0,step=4.0f0)]

# ╔═╡ 7f3602e9-028f-44fc-b7dd-052f76438dae
full_data = map(MLNanoShaper.implicit_surface.(Ref(atoms), models[1:1], param[1:1])) do (points, top)
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
data = select_in_domaine.(((x, _, z),) -> -5 <= x <= 5, full_data)

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
_ref = load("$(homedir())/datasets/pqr/$prot_num/triangulatedSurf.off")

# ╔═╡ eec7338d-a319-478c-9cde-663e38b3e523
ref = Ms.SimpleMesh(coordinates(_ref) .|> Tuple, GeometryBasics.faces(_ref) .|> Tuple .|> Ms.connect)

# ╔═╡ e7e6584b-5059-46f6-a614-76866f1b1df9
begin
    f = Mk.Figure(size = (1000,700))
    Mk.Axis3(f[1, 1], title="tiny_angular_dense_cv 3A")
    Ms.viz!(f[1, 1], meshes[1]; color=:red)
    Mk.Axis3(f[1, 2], title="tiny_angular_dense_cv 2A")
    Ms.viz!(f[1, 2], meshes[2]; color=:red)
    Mk.Axis3(f[2, 1], title="tiny_angular_dense_cv 3A")
    Ms.viz!(f[2, 1], full_meshes[1]; color=:red)
    Ms.viz!(f[2, 1], ref; color=:green)
    a = Mk.Axis3(f[2, 2], title="tiny_angular_dense_cv 2A")
    l = Ms.viz!(f[2, 2], ref; color=:green)
    Ms.viz!(f[2, 2], full_meshes[2]; color=:red)
	Mk.Legend(f[1:2,3],[Mk.LineElement(color = :green),Mk.LineElement(color = :red)],["true value","predicted value"])
	f
end

# ╔═╡ e78e5812-1927-4f67-bd3a-9bd1b577f9ad
function get_input_slice(atoms,step,z)
	(; mins, maxes) = atoms.tree.hyper_rec
    ranges = range.(mins, maxes; step)
    grid = Point3f.(reshape(ranges[1], :, 1), reshape(ranges[2], 1, :),z)
end

# ╔═╡ 2b0fc2fd-47c1-491c-b9a3-6ddff7b61850
function get_slice(atoms,model,z,(;cutoff_radius,step,default_value))
    grid = get_input_slice(atoms,step,z)
	volume = Folds.map(grid) do x
        evaluate_model(model, x, atoms; cutoff_radius,default_value)
    end
end

# ╔═╡ ee6dc376-b884-4ecb-8c63-1830bd664597
slice1 = get_slice(atoms,models[1],6.0,(;cutoff_radius=3.0f0,step=.1f0,default_value=-10f0))

# ╔═╡ a8a43fdf-f69c-41ef-b309-ee8531e5df23
slice2 = get_slice(atoms,models[2],6.0,(;cutoff_radius=2.0f0,step=.1f0,default_value=0.0f0))

# ╔═╡ d679ca88-615e-4675-9d0a-419cd18246f9
begin
    g = Mk.Figure(size = (1200,500))
    Mk.Axis(g[1, 1], title="tiny_angular_dense_cv 3A")
    plt1 = Mk.plot!(g[1, 1], exp.(slice1) ./(exp.(-slice1) .+ exp.(slice1));colormap = :rainbow,colorrange = [0,1])
	Mk.Colorbar(g[1, 2],plt1)
    Mk.Axis(g[1, 3], title="tiny_angular_dense_cv 2A")
    plt2 = Mk.plot!(g[1, 3], slice2;colormap = :rainbow,colorrange = [0,1])
	Mk.Colorbar(g[1, 4],plt2)
	g
end

# ╔═╡ 399eb8e8-0109-4cdd-887e-d1456bf72729
(; mins, maxes) = atoms.tree.hyper_rec

# ╔═╡ 67cb1851-a1a3-458f-9c9d-b5061a57ed37
ranges = range.(mins, maxes; step=.1)

# ╔═╡ 72e3bde9-6be8-46ac-899d-27afd99a7b3e
 grid = get_input_slice(atoms,.1,6)

# ╔═╡ a98327b5-f8a4-46cc-a14c-8dd234ca9933
dist = signed_distance.(grid,Ref(RegionMesh(surface)))

# ╔═╡ 44e41f3b-69c5-47f5-bb2e-b1d668eb2889
begin
	h = Mk.Figure(size = (600,500))
	Mk.Axis(h[1, 1], title="tiny_angular_dense_cv 3A")
	Mk.contour!(h[1,1],ranges[1],ranges[2],slice1,levels=[.5])
	Mk.contour!(h[1,1],ranges[1],ranges[2],dist,levels=[0])
	h
end

# ╔═╡ Cell order:
# ╟─5f801ac4-1f27-11ef-3246-afece906b714
# ╠═8cbe5092-8af4-4a2c-a574-e33cb9632058
# ╠═9bbec073-a5bf-429f-bc29-7411e3041888
# ╠═ea5ad13f-bf1f-4986-b781-a30a8ca83e7c
# ╠═fc935a86-ceac-4d5a-8fcb-34d9c754a2f1
# ╠═e8a48b40-28b8-41a7-a67d-cbac2b361f84
# ╠═ccbcea27-ea65-4b0c-8a56-c3a21fc976bb
# ╠═69ee1b79-b99d-4e3a-9769-254b1939aba6
# ╠═f7041ca8-97be-4998-9c10-2cbed79eb135
# ╠═58cf0ac8-d68d-47a7-b08f-098b65d19908
# ╠═a0a5f16f-0224-47b1-ae86-c4b5bd48fd07
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
# ╠═e78e5812-1927-4f67-bd3a-9bd1b577f9ad
# ╠═2b0fc2fd-47c1-491c-b9a3-6ddff7b61850
# ╠═ee6dc376-b884-4ecb-8c63-1830bd664597
# ╠═a8a43fdf-f69c-41ef-b309-ee8531e5df23
# ╠═d679ca88-615e-4675-9d0a-419cd18246f9
# ╠═399eb8e8-0109-4cdd-887e-d1456bf72729
# ╠═67cb1851-a1a3-458f-9c9d-b5061a57ed37
# ╠═72e3bde9-6be8-46ac-899d-27afd99a7b3e
# ╠═a98327b5-f8a4-46cc-a14c-8dd234ca9933
# ╠═44e41f3b-69c5-47f5-bb2e-b1d668eb2889
