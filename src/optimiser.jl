using Optimisers: @.., _eps,  @def 

"""
    AdamDelta(ρ = 0.9, ϵ = 1e-8)
    AdaDelta(; [rho, epsilon])


# Parameters
- Rho (`ρ == rho`): Factor by which the gradient is decayed at each time step.
- Machine epsilon (`ϵ == epsilon`): Constant to prevent division by zero
                         (no need to change default)
"""
@def struct AdamDelta <: AbstractRule
  rho = 0.99
  beta = 0.99
  epsilon = 1e-8
end

Optimisers.init(o::AdamDelta, x::AbstractArray) = (zero(x),zero(x),zero(x),eltype(x)(o.beta))

function Optimisers.apply!(o::AdamDelta, state, _::AbstractArray{T}, dx) where T
  ρ,β, ϵ = T(o.rho),T(o.beta), _eps(T, o.epsilon)
  acc, Δacc,dacc, beta_t = state

  @.. acc = ρ * acc + (1 - ρ) * abs2(dx)
  @.. dacc= β* dacc + (1 - β) * dx
  dx′ = @. dacc * sqrt((Δacc + ϵ) /(acc + ϵ)) / (1 - beta_t)  # Cannot be lazy as this needs the old Δacc
  @.. Δacc = ρ * Δacc + (1 - ρ) * abs2(dx′)

  return (acc, Δacc, dacc,beta_t * β), dx′
end
