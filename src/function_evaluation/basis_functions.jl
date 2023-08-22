Base.@kwdef struct Parameters{P,Θ}
    θ::Θ
    p::P = nothing
end

"""
    BasisSet{F<:Tuple} <: Function

A set of basis functions. 

# Fields 
- `bases::F`: A `Tuple` of basis functions, each function being of the form `(q, p) -> Number`, with the same `p` for each basis function.

# Constructors

    BasisSet((f1, f2, ...))
    BasisSet(f1, f2, ...)

# Evaluating 

You can evaluate a `BasisSet` using the method

    (f::BasisSet{F})(q, θ, p) where {F} 

which returns 

```math
\\sum_{i=1}^n \\theta_if_i(q, p)
```
"""
struct BasisSet{F<:Tuple} <: Function
    bases::F
end
@inline BasisSet(bases...) = BasisSet(bases)
Base.length(f::BasisSet) = length(f.bases)
Base.eachindex(f::BasisSet) = eachindex(f.bases)
BasisSet() = BasisSet(())

"""
    PolynomialBasis(d1, d2)

Construct a set of polynomial basis functions of degree `d1` to `d2`, 
returning a `BasisSet` object.

# Examples 
```jldoctest 
julia> basis = PolynomialBasis(2, 4);

julia> basis(0.5, [0.1, 0.67, -2.3], nothing)
-0.034999999999999976

julia> 0.1 * 0.5^2 + 0.67 * 0.5^3 - 2.3 * 0.5^4
-0.034999999999999976
```
"""
function PolynomialBasis(d1, d2)
    n1, n2 = min(d1, d2), max(d1, d2)
    bases = ntuple(let d1 = n1
            i -> let d = d1 + i - 1
                (u, p) -> u^d
            end
        end, n2 - n1 + 1)
    if d1 ≤ d2 
        return BasisSet(bases)
    else
        return BasisSet(reverse(bases))
    end
end

@inline function (f::BasisSet{F})(u, θ::AbstractVector, p) where {F}
    return _eval_basis_set(f.bases, u, θ, p)
end
@inline (f::BasisSet{<:Tuple})(u, x, t, p::Parameters) = f(u, p.θ, p.p)
@inline (f::BasisSet{<:Tuple})(u, t, p::Parameters) = f(u, p.θ, p.p)
@unroll function _eval_basis_set(bases::F, u, θ, p) where {F<:Tuple}
    s = zero(u)
    @unroll for i in 1:length(bases)
        φ = bases[i]
        θᵢ = θ[i]
        s = muladd(θᵢ, φ(u, p), s)
    end
    return s
end

@inline function eval_and_diff(f::BasisSet{F}, u, p, idx::Integer) where {F}
    return _eval_and_diff_basis_function(f.bases[idx], u, p)
end
function _eval_and_diff_basis_function(φ, u, p)
    g = let ρ = p
        s -> φ(s, ρ)
    end
    result = DiffResults.DiffResult(u, u)
    result = ForwardDiff.derivative!(result, g, u)
    gu = DiffResults.value(result)
    ∂gu = DiffResults.derivative(result)
    return gu, ∂gu
end

function eval_and_diff(f::BasisSet{F}, u, θ, p) where {F}
    s = zero(u)
    s′ = zero(u)
    for i in 1:length(f)
        φ = f.bases[i]
        θᵢ = θ[i]
        φᵢ, φᵢ′ = _eval_and_diff_basis_function(φ, u, p)
        s = muladd(θᵢ, φᵢ, s)
        s′ = muladd(θᵢ, φᵢ′, s′)
    end
    return s, s′
end