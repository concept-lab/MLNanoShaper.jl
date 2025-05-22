GlobalPreprocessed = @NamedTuple{
    inputs::ConcatenatedBatch{T,Int,Vector{Int}},
    d_reals::Vector{Float32}
} where {T <: AbstractArray{Float32}}

function loggit(x)
    log(x) - log(1 - x)
end

function KL(true_probabilities::AbstractArray{T},
        expected_probabilities::AbstractArray{T}) where {T <: Number}
    epsilon = 1.0f-5
    sum(
        true_probabilities .*
        log.((true_probabilities .+ T(epsilon)) ./ (expected_probabilities .+ T(epsilon))),
        dims = 1)
end

struct BayesianStats
    nb_true_positives::Int
    nb_false_negatives::Int
    nb_true::Int
    nb_false::Int
    function BayesianStats(
            nb_true_positives::Int, nb_false_negatives::Int, nb_true::Int, nb_false::Int)
        @assert nb_true_positives<=nb_true "got $nb_true_positives true positives for $nb_true true values"
        @assert nb_false_negatives<=nb_false "got $nb_false_negatives true negatives for $nb_false true values"
        new(nb_true_positives, nb_false_negatives, nb_true, nb_false)
    end
end
function BayesianStats(real::AbstractVector{Bool}, pred::AbstractVector{Bool})
    nb_true_positives = count(real .&& pred)
    nb_false_negatives = count(.!real .&& pred)
    nb_true = count(real)
    nb_false = count(.!real)
    ignore_derivatives() do
        @debug "statistics" nb_true nb_false nb_true_positives nb_false_negatives
    end
    BayesianStats(nb_true_positives, nb_false_negatives, nb_true, nb_false)
end
function reduce_stats((;
        nb_true_positives, nb_false_negatives, nb_true, nb_false)::BayesianStats)
    ignore_derivatives() do
        @debug "reduce" nb_true nb_false nb_true_positives nb_false_negatives
    end

    false_positive_rate = 1 - nb_true_positives / nb_true
    false_negative_rate = nb_false_negatives / nb_false
    error_rate = max(false_positive_rate, false_negative_rate)
    (; false_positive_rate, false_negative_rate, error_rate)
end

function aggregate(x::AbstractArray{BayesianStats})
    nb_true_positives = sum(getproperty.(x, :nb_true_positives))
    nb_false_negatives = sum(getproperty.(x, :nb_false_negatives))
    nb_true = sum(getproperty.(x, :nb_true))
    nb_false = sum(getproperty.(x, :nb_false))
    BayesianStats(nb_true_positives, nb_false_negatives, nb_true, nb_false) |> reduce_stats
end
aggregate(x::AbstractArray{<:Number}) = mean(x)
function aggregate(x::StructArray)
    Dict(keys(StructArrays.components(x)) .=>
        aggregate.(values(StructArrays.components(x))))
end
function aggregate(w::AbstractDict{<:Any, <:AbstractVector})
    r = Dict(
        keys(w) .=> aggregate.(values(w)))
    r[:global] = aggregate(reduce(vcat, values(w)))
    r
end
function aggregate(x::Any)
    error("not expected $x of type $(typeof(x))")
end

"""
	abstract type LossType end
LossType is an interface for defining loss functions.
# Implementation
- get_loss_fn(::LossType)::Function : the associated loss function
- _metric_type(::Type{<:LossType)::Type : the type of metrics returned by the loss function
- _get_loss_type(::StaticSymbol)::LossType : the function generating the loss_type
"""
abstract type LossType end
_metric_type(x::Type{<:LossType}) = error("_metric_type not implemented for $x")
get_loss_fn(x::LossType) = error("get_loss_fn not implemented for $x")
_get_loss_type(x::StaticSymbol) = error("_get_loss_type not implemented for $x")

metric_type(x::LossType) = _metric_type(typeof(x))
metric_type(x::Type{<:LossType}) = _metric_type(x)

get_loss_type(x::Symbol) = _get_loss_type(static(x))
get_loss_type(x::StaticSymbol) = _get_loss_type(x)

CategoricalMetric = @NamedTuple{
    stats::BayesianStats,
    kl_div::Float32,
    outer_reg_loss::Float32,
}
function generate_true_probabilities(d_real::AbstractArray)
    epsilon = 1.0f-5
    is_inside = d_real .> epsilon
    is_outside = d_real .< epsilon
    is_surface = abs.(d_real) .<= epsilon
    probabilities = zeros32(2, length(d_real))
    probabilities[1, :] .= (1 - epsilon) * is_inside + 1 / 2 * is_surface +
                           epsilon * is_outside
    probabilities[2, :] .= (1 - epsilon) * is_outside + 1 / 2 * is_surface +
                           epsilon * is_inside
    probabilities
end
function max_abs_weight(ps)
  max_val = 0.0  # Initialize with a value that will be smaller than any absolute weight

  function traverse(x)
    if x isa AbstractArray
      max_val = max(max_val, maximum(abs.(x)))
    elseif x isa NamedTuple || x isa Tuple
      foreach(traverse, values(x))
    end
  end

  traverse(ps)
  return max_val
end


function get_inner_regularisation_loss(model::AbstractLuxLayer,ps,st,input)
    _input = ignore_derivatives() do
        device = model[2].prepross[1].func
        a = similar(input.field,6,1)
        a[:,1] .= eltype(a).([1,2,0,0,0,1])
        ConcatenatedBatch(a,[0,1]) |> device
    end
    output = first(model(_input,ps,st))
    sum(abs.(output .- one(eltype(input.field))))
end

function get_outer_regularisation_loss(model::AbstractLuxLayer,ps,st,input)
    intermediary_model = ignore_derivatives() do
         get_last_chain(model)
     end
    input = ignore_derivatives() do
        device = model[2].prepross[1].func
        a = similar(input.field,get_last_chain_dim(model),1) |> device
        a .= zero(eltype(input.field))
        # @info "input" a input.field
        a
    end
    output = first(intermediary_model(input,ps,st))
    sum(abs.(output))
end


"""
    categorical_loss(model, ps, st, (; point, atoms, d_real))

The loss function used by in training.
Return the KL divergence between true probability and empirical probability
Return the error with the espected distance as a metric.
"""
function categorical_loss(model::Lux.AbstractLuxLayer,
        ps,
        st,
        (;
            inputs,
            d_reals))::Tuple{Float32, <:Any, CategoricalMetric}
    # ignore_derivatives() do
        # @info "loss" model inputs ps st
    # end
    v_pred, _st = model(inputs, ps, st)
    epsilon = 1.0f-5
    v_pred = epsilon .+ v_pred .*(1-2*epsilon)
    v_pred = vcat(v_pred, 1 .- v_pred)
    v_pred = cpu_device()(v_pred)
    probabilities = ignore_derivatives() do
        generate_true_probabilities(d_reals)
    end 
    model,st,inputs = ignore_derivatives() do
        model,st,inputs
    end
    kl_div =  mean(KL(probabilities, v_pred))
    outer_reg_loss = get_outer_regularisation_loss(model,ps,st,inputs)
    r1, r2 = ignore_derivatives() do
        ((kl_div,outer_reg_loss) .+ epsilon) ./ (outer_reg_loss + kl_div +2*epsilon) 
    end
    loss = r1 * kl_div + r2 * outer_reg_loss
    stats = ignore_derivatives() do
        true_vec = Iterators.filter(vec(d_reals)) do dist
            abs(dist) > epsilon
        end .> 0
        selected = Iterators.filter(zip(d_reals, vec(v_pred[1, :]))) do (dist, _)
            abs(dist) > epsilon
        end
        pred_vec = map(selected) do (_, pred)
            pred > 0.5f0
        end
        BayesianStats(true_vec, pred_vec)
    end    

    (loss, _st, (;stats,kl_div,outer_reg_loss))
end

struct CategoricalLoss <: LossType end
_metric_type(::Type{CategoricalLoss}) = CategoricalMetric
get_loss_fn(::CategoricalLoss) = categorical_loss
_get_loss_type(::StaticSymbol{:categorical}) = CategoricalLoss()

ContinousMetric = @NamedTuple{
    stats::BayesianStats,
    bias_error::Float32,
    abs_error::Float32,
    bias_distance::Float32,
    abs_distance::Float32}

"""
    continus_loss(model, ps, st, (; point, atoms, d_real))

The loss function used by in training.
compare the predicted (square) distance with \$\\frac{1 + \tanh(d)}{2}\$
Return the error with the espected distance as a metric.
"""
function continus_loss(model,
        ps,
        st,
        (;
            inputs,
            d_reals))::Tuple{Float32, Any, ContinousMetric}
    v_pred, st = model(inputs, ps, st)
    v_pred = cpu_device()(v_pred)
    v_real = σ.(d_reals)
    error = v_pred .- v_real
    loss = mean(error .^ 2)
    D_distance = loggit.(max.(0, v_pred) * (1 .- 1.0f-4)) .- d_reals

    epsilon = 1.0f-5
    true_vec = Iterators.filter(vec(d_reals)) do dist
        abs(dist) > epsilon
    end .> 0.5f0
    pred_vec = map(Iterators.filter(zip(d_reals, vec(v_pred))) do (dist, _)
        abs(dist) > epsilon
    end) do (_, pred)
        pred > 0.5f0
    end
    (loss,
        st,
        (; stats = BayesianStats(true_vec, pred_vec),
            bias_error = mean(error),
            abs_error = mean(abs.(error)),
            bias_distance = mean(D_distance),
            abs_distance = abs.(D_distance) |> mean))
end

struct ContinousLoss <: LossType end
_metric_type(::Type{ContinousLoss}) = ContinousMetric
get_loss_fn(::ContinousLoss) = continus_loss
_get_loss_type(::StaticSymbol{:continuous}) = ContinousLoss()
