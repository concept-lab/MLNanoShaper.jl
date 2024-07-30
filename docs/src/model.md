# Building a new model

In order to create a new model, we need to create a new function that returns a `Lux.Abstractlayer` in the `MLNanoShaperModule`.
The function must take as input at least a name, `van_der_waals_channel`, `on_gpu`,and `cutoff_radius`.

```julia
function custom_angular_dense(; name::String,
        van_der_waals_channel = false, on_gpu = true, cutoff_radius::Float32 = 3.0f0)
    main_chain = Parallel(.*,
            Chain(Dense(6 => 10, elu),
                Dense(10 => 5, elu)),
            Lux.WrappedFunction(scale_factor)
    )
    main_chain = DeepSet(Chain(
        symetrise(; cutoff_radius, device = on_gpu ? gpu_device() : identity),
        main_chain
    ))
    secondary_chain = Chain(
            BatchNorm(5),
            Dense(5  => 10, elu),
            Dense(10 => 1, sigmoid_fast));
    Chain(PreprocessingLayer(Partial(select_and_preprocess; cutoff_radius)),
        main_chain,
        secondary_chain;
        name)
end
```

Once this is done you can call the model by using the flag `--model` with the name of the function created. In our case
`--model custom_angular_dense`.
