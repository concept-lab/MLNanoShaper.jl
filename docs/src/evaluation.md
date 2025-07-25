# Model evaluation
Model are stored as serialized objects containing the parameters, weights, and a partial application to evaluate the model
( it is done this way to prevent serialization of anonymous functions that behave badly with serialization).
Given a path `path`, we first deserialize de model to get a SerializedModel and turn it into a `Lux` `StatefullLuxLayer` with production_instantiate.
Since preprocessing is part of the model, the input is alway on the cpu but the output might be on gpu and  must be moved to cpu with `cpu_device()`. 
To evaluate the model, starting from a vector of spheres,you will use `RegularGrid` to create a regular grid.
To evaluate points, they must be dispatched within a batch.
Here is an example of evaluation.
```julia
function evaluate_field(model_path::AbstractString,atoms::Vector{Sphere{Float32}};step::Number=1,batch_size = 100000)::Array{Float32,3}
  model = production_instantiate(deserialize(model_path))
  atoms = RegularGrid(aatoms,get_cutoff_radius(model.model))
	mins = atoms.start .- 2
	maxes = mins .+ size(atoms.grid) .* atoms.radius .+ 2
    ranges = range.(mins, maxes; step)
    grid = Point3f.(reshape(ranges[1], :, 1,1), reshape(ranges[2], 1, :,1), reshape(ranges[3], 1,1,:))
    g = vec(grid)
    volume = similar(grid,Float32)
    v = reshape(volume,:)
    # @info "comparing lengths" length(volume)/batch_size 
    for i in 1:batch_size:length(volume)
    	k = min(i+ batch_size-1,length(v))
    	res =  model((Batch(view(g,i:k)), atoms)) |> cpu_device() |> vec
    	v[i:k] .= res
        end
	volume
end
```

