using NearestNeighbors
using GeometryBasics
using LinearAlgebra
using Base.Iterators

struct RegionMesh
	triangles::Vector{TriangleFace{Point3f}}
    tree::KDTree
end
function RegionMesh(mesh::GeometryBasics.Mesh)
	triangles = map(eachindex(coordinates(mesh))) do i
		j,_ = Iterators.filter(enumerate(faces(mesh))) do (_,tri)
			OffsetInteger{-1, UInt32}(i) in tri
		end |> first
		map(faces(mesh)[j]) do j
			coordinates(mesh)[j]
		end |> TriangleFace{Point3f}
	end
	
    RegionMesh(triangles,
        KDTree(coordinates(mesh); reorder = false))
end

distance(x::Point3, y::KDTree) = nn(y, x) |> last

function signed_distance(p::Point3, mesh::RegionMesh)
    id_point, dist = nn(mesh.tree,p)
    x, y, z = mesh.triangles[OffsetInteger{-1, UInt32}(id_point)]
    # @info "triangle" x y z

    direction = hcat(y - x, z - x, p - x) |> det |> sign
    direction #* dist
end

nograd(f, args...; kargs...) = f(args...; kargs...)

function ChainRulesCore.rrule(::typeof(nograd), f, args...; kargs...)
    res = f(args...; kargs...)
    function knn_pullback(_)
        tuple(fill(NoTangent(), length(args)))
    end
    res, knn_pullback
end
