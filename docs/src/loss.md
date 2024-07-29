```@meta
CurrentModule = MLNanoShaper
```


# Building custom loss functions
A loss function is a function that take as input a lux layer, the parameters and state, a named tuple containing the points where the model is evaluated, the preprocessed input and the algebric distance to the surface.
The loss function then return 2 values:
- the loss : a scalar number
- the state of the model
- a nambed tuple containing evaluations metrics
```
function custom_loss(model,
        ps,
        st,
        (; points,
            inputs,
            d_reals))::Tuple{
        Float32, Any, CategoricalMetric}
    # model evaluation
    v_pred, st = Lux.apply(model, inputs, ps, st)
    v_pred = vcat(v_pred, 1 .- v_pred)
    v_pred = cpu_device()(v_pred)
    probabilities = ignore_derivatives() do
        generate_true_probabilities(d_reals)
    end
    (KL(probabilities, v_pred) |> mean,
        st, (;))
end
```
Once we have the loss function we need to register it in order to use in at the command line level.

First we need a type to represent the loss function.
```
struct CustomLoss <: LossType end
```
Then we need to give the type of metric used by the model. In our case it is a empty `NamedTuple`.
```
_metric_type(::Type{CustomLoss}) = @NamedTuple{}
```
We need to associate the loss function to our new type.
```
get_loss_fn(::CustomLoss) = custom_loss
```
At the end we need to give the name that will be used at the command line level to select our loss.
```
_get_loss_type(::StaticSymbol{:custom}) = CustomLoss()
```
