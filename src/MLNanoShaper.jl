module MLNanoShaper
export extract_balls

using StructArrays
using GLMakie
using GeometryBasics
using BioStructures

struct XYZR{T} end
struct DICT{T} end

function read_sphere(io::IO, ::Type{Sphere{T}}) where {T}
    line = readline(io)
    x, y, z, r = parse.(T, split(line))
    Sphere(Point3(x, y, z), r)
end
function Base.read(io::IO, ::XYZR{T}) where {T}
    out = Sphere{T}[]
    while !eof(io)
        push!(out, read_sphere(io, Sphere{T}))
    end
    out
end

function viz(x::AbstractVector{Sphere{T}}) where {T}
    fig = Figure()
    ax = Axis3(fig[1, 1])
    mesh!.(Ref(ax), x)
    fig
end

repeat(f, n::Integer) = foldr(∘, Iterators.repeated(f, n))
function extract_balls(prot::ProteinStructure)
    recurse(fun, arg) = mapreduce(vcat, fun, arg)
    repeat(x -> recurse(atom -> Point3(atom.coord...), x), 4)(prot)
end
end
