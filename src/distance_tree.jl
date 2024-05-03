using RegionTrees
using NearestNeighbors
using GeometryBasics
using LinearAlgebra
using Base.Iterators


struct RegionMesh
	mesh::GeometryBasics.Mesh
	tree::KDTree
end
distance(x::Point3,y::KDTree) = nn(y,x) |> last

function signed_distance(p::Point3,y::RegionMesh)
	id_point, dist = distance(p,y.tree)
	@info id_point
	id_triangle = Iterators.filter(GeometryBasics.faces(y.mesh)) do id_triangle
		OffsetInteger{-1,UInt32}(id_point) in GeometryBasics.faces(y.mesh)[id_triangle]
	end |> first
	x,y,z = map(i -> coordinates(y.mesh)[i],GeometryBasics.faces(y.mesh)[id_triangle])
	# @info "triangle" x y z

	direction = hcat(y -x, z- x,p -x) |>det |> sign
	direction #* d
end

nograd(f,args...;kargs...) = f(args...;kargs...)

function ChainRulesCore.rrule(::typeof(nograd),f,args...;kargs...)
	res = f(args...;kargs...)
	function knn_pullback(_)
		tuple(fill(NoTangent(),length(args)))
	end
	res,knn_pullback
end
