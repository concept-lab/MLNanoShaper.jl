module MLNanoShaper

using StructArrays
using GeometryBasics: Sphere
using GLMakie


function Base.read(io::IO, ::Type{Sphere{T}}) where {T}
	line = readline(io)
	x,y,z,r = parse.(T,split(line))
	Sphere(Point3(x,y,z),r)
end
function Base.read(io::IO, ::Type{Vector{Sphere{T}}}) where {T}
	out = Sphere{T}[]
	while !eof(io)
		push!(out,read(io, Sphere{T}))
	end
	out
end

function viz(x::AbstractVector)
	fig  = Figure()
	ax = Axis(fig[1,1])
	mesh!.(Ref(ax),x)
	fig
end

end
