
using StructArrays
using GLMakie
using GeometryBasics
using BioStructures
export extract_balls
struct XYZR{T} end
struct DICT{T} end

function read_line(io::IO, ::Type{XYZR{T}}) where {T}
    line = readline(io)
    x, y, z, r = parse.(T, split(line))
    Sphere(Point3(x, y, z), r)
end
function Base.read(io::IO, ::Type{XYZR{T}}) where {T}
    out = Sphere{T}[]
    while !eof(io)
        push!(out, read_line(io, XYZR{T}))
    end
	Set(out)
end

function read_line(io::IO, ::Type{DICT{T}}) where {T}
    line = readline(io)
    x, y = split(line)
    String(x) => parse(T, y)
end
function Base.read(io::IO, ::Type{DICT{T}}) where {T}
    out = Dict{String, T}()
    while !eof(io)
        push!(out, read_line(io, DICT{T}))
    end
    out
end
function viz(x::AbstractSet{Sphere{T}}) where {T}
    fig = Figure()
    ax = Axis3(fig[1, 1])
    mesh!.(Ref(ax), x)
    fig
end

reduce(fun, arg) = mapreduce(fun, vcat, arg)
reduce(fun) = arg -> reduce(fun, arg)
function reduce(fun, arg, n::Integer)
    if n <= 1
        reduce(fun, arg)
    else
        reduce(reduce(fun), arg, n - 1)
    end
end

function Base.print(io::IO, prot::AbstractSet{Sphere{T}}, ::Type{XYZR{T}}) where {T}
	for sph in prot
		println(io,sph.center[1]," ",sph.center[2]," ",sph.center[3]," ",sph.r)
	end
end

function extract_balls(prot::ProteinStructure,
        radii::Dict{String, Float64})::Set{Sphere{Float64}}
    reduce(prot, 4) do atom
        if typeof(atom) == Atom
            [Sphere(Point3(atom.coords), if atom.element in keys(radii)
                radii[atom.element]
            else
                1.0
            end)]
        else
            []
        end
    end |> Set
end
