
"""
	accumulator(processing,logger)
A processing logger that transform logger on multiple batches
Ca be used to liss numerical data, for logging to TensorBoardLogger.
"""
mutable struct AccumulatorLogger <: AbstractLogger
    processing::Function
    logger::AbstractLogger
    data::Dict
end

function Logging.handle_message(logger::AccumulatorLogger, message...; kargs...)
    logger.processing(logger.logger, Ref(logger.data), message, kargs)
end
Logging.shouldlog(logger::AccumulatorLogger, args...) = shouldlog(logger.logger, args...)
Logging.min_enabled_level(logger) = Logging.min_enabled_level(logger.logger)

empty_accumulator(::Union{Type{<:AbstractDict}, Type{<:NamedTuple}}) = Dict()
empty_accumulator(T::Type{<:Number}) = T[]
empty_accumulator(::Type{T}) where {T} = Ref{T}()

function accumulate(d::Dict, kargs::AbstractDict)
    for k in keys(kargs)
        if k ∉ keys(d)
            d[k] = empty_accumulator(typeof(kargs[k]))
        end
        accumulate(d[k], kargs[k])
    end
end
function accumulate(d::Vector{T}, arg::T) where {T <: Number}
    push!(d, arg)
end
function accumulate(d::Dict, kargs::NamedTuple)
    accumulate(d, pairs(kargs))
end
function accumulate(d::Ref, arg)
    d[] = arg
end

extract(d::Dict) = Dict(keys(d) .=> extract.(values(d)))
function extract(d::Vector{<:Number})::Number
    mean(d)
end
extract(d::Ref) = d[]

function get_logger(logdir::String,epoch::Int)::AbstractLogger
    # logger = TeeLogger(TBLogger(loggdir), global_logger())
    # logger = AccumulatorLogger(logger,
    #     Dict()) do logger, d, args, kargs
    #     level, message = args
    #     if message == "epoch"
    #         kargs = extract(d)
    #         for (k, v) in pairs(kargs)
    #             Logging.handle_message(logger, level, k, args[3:end]...; v...)
    #         end
    #         d[] = Dict()
    #     else
    #         accumulate(d[], Dict([message => kargs]))
    #     end
    # end
    logger = TBLogger(logdir,tb_append,purge_step=epoch)
    TeeLogger(ActiveFilteredLogger(logger) do (; message)
            message in ("log",)
        end,
        ActiveFilteredLogger(global_logger()) do (; message)
            message ∉ ("log",)
        end)
end
