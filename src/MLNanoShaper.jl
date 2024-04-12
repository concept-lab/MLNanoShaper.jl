module MLNanoShaper

using StructArrays
using GeometryBasics: Sphere
using GLMakie

function Base.read(io::IO, ::Type{Sphere{T}}) where {T}
	Sphere([read(io, T), read(io, T), read(io, T)], read(io, T))
end
function Base.read(io::IO, ::Type{StructArray{Sphere{T}}}) where {T}
    StructArray{Sphere}(read.(io, Sphere{T}))
end

mesh(Sphere(Point3(0,0,0),1))



end
