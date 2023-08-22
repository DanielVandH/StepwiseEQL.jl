function forward_dfdx(f₁, f₂, f₃, x₁, x₂, x₃) # f'(x₁)
    return (1 / (x₁ - x₂) + 1 / (x₁ - x₃)) * f₁ - (x₁ - x₃) / ((x₁ - x₂) * (x₂ - x₃)) * f₂ + (x₁ - x₂) / ((x₁ - x₃) * (x₂ - x₃)) * f₃
end
function central_dfdx(f₁, f₂, f₃, x₁, x₂, x₃) # f'(x₂)
    return (x₂ - x₃) / ((x₁ - x₂) * (x₁ - x₃)) * f₁ + (1 / (x₂ - x₃) - 1 / (x₁ - x₂)) * f₂ + (x₂ - x₁) / ((x₁ - x₃) * (x₂ - x₃)) * f₃
end
function backward_dfdx(f₁, f₂, f₃, x₁, x₂, x₃) # f'(x₃)
    return (x₃ - x₂) / ((x₁ - x₂) * (x₁ - x₃)) * f₁ + (x₁ - x₃) / ((x₁ - x₂) * (x₂ - x₃)) * f₂ - (1 / (x₁ - x₃) + 1 / (x₂ - x₃)) * f₃
end
function forward_d²fdx²(f₁, f₂, f₃, x₁, x₂, x₃) # f''(x₁)
    return 2 / ((x₁ - x₂) * (x₁ - x₃)) * f₁ - 2 / ((x₁ - x₂) * (x₂ - x₃)) * f₂ + 2 / ((x₁ - x₃) * (x₂ - x₃)) * f₃
end
function central_d²fdx²(f₁, f₂, f₃, x₁, x₂, x₃) # f''(x₂)
    return 2 / ((x₁ - x₂) * (x₁ - x₃)) * f₁ - 2 / ((x₁ - x₂) * (x₂ - x₃)) * f₂ + 2 / ((x₁ - x₃) * (x₂ - x₃)) * f₃
end
function backward_d²fdx²(f₁, f₂, f₃, x₁, x₂, x₃) # f''(x₃)
    return 2 / ((x₁ - x₂) * (x₁ - x₃)) * f₁ - 2 / ((x₁ - x₂) * (x₂ - x₃)) * f₂ + 2 / ((x₁ - x₃) * (x₂ - x₃)) * f₃
end

function cell_density(r::AbstractVector, x::Number, iguess=0)
    idx = max(1, min(DataInterpolations.searchsortedlastcorrelated(r, x, iguess), length(r) - 1))
    q₁ = cell_density(r, idx)
    q₂ = cell_density(r, idx + 1)
    x₁ = r[idx]
    x₂ = r[idx+1]
    θ = (x - x₁) / (x₂ - x₁)
    q = (1 - θ) * q₁ + θ * q₂
    return max(q, zero(q)), idx
end
function cell_density(r::AbstractVector, i::Integer)
    if i == firstindex(r)
        #return inv(r[begin+1] - r[begin])
        return 2 / (r[begin+1] - r[begin]) - 2 / (r[begin+2] - r[begin])
    elseif i == lastindex(r)
        #return inv(r[end] - r[end-1])
        return 2 / (r[end] - r[end-1]) - 2 / (r[end] - r[end-2])
    else
        return 2inv(r[i+1] - r[i-1])
    end
end
function cell_density(cell_sol::ODESolution, i, j)
    rʲ = cell_sol.u[j]
    qᵢʲ = cell_density(rʲ, i)
    return qᵢʲ
end
function cell_density(cell_sol::AveragedODESolution, i, j)
    return cell_sol.q[j][i]
end

function cell_∂q∂x(cell_sol, i, j)
    rʲ = cell_sol.u[j]
    qᵢʲ = cell_density(cell_sol, i, j)
    xᵢʲ = rʲ[i]
    if i == firstindex(rʲ)
        qᵢ₊₁ʲ = cell_density(cell_sol, i + 1, j)
        qᵢ₊₂ʲ = cell_density(cell_sol, i + 2, j)
        xᵢ₊₁ʲ = rʲ[i+1]
        xᵢ₊₂ʲ = rʲ[i+2]
        return (qᵢ₊₁ʲ - qᵢʲ) / (xᵢ₊₁ʲ - xᵢʲ)
        ∂q∂x = forward_dfdx(qᵢʲ, qᵢ₊₁ʲ, qᵢ₊₂ʲ, xᵢʲ, xᵢ₊₁ʲ, xᵢ₊₂ʲ)
    elseif i == lastindex(rʲ)
        qᵢ₋₂ʲ = cell_density(cell_sol, i - 2, j)
        qᵢ₋₁ʲ = cell_density(cell_sol, i - 1, j)
        xᵢ₋₂ʲ = rʲ[i-2]
        xᵢ₋₁ʲ = rʲ[i-1]
        return (qᵢʲ - qᵢ₋₁ʲ) / (xᵢʲ - xᵢ₋₁ʲ)
        ∂q∂x = backward_dfdx(qᵢ₋₂ʲ, qᵢ₋₁ʲ, qᵢʲ, xᵢ₋₂ʲ, xᵢ₋₁ʲ, xᵢʲ)
    else
        qᵢ₋₁ʲ = cell_density(cell_sol, i - 1, j)
        qᵢ₊₁ʲ = cell_density(cell_sol, i + 1, j)
        xᵢ₋₁ʲ = rʲ[i-1]
        xᵢ₊₁ʲ = rʲ[i+1]
        ∂q∂x = central_dfdx(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, xᵢ₋₁ʲ, xᵢʲ, xᵢ₊₁ʲ)
    end
    return ∂q∂x
end

function cell_∂²q∂x²(cell_sol, i, j)
    qᵢʲ = cell_density(cell_sol, i, j)
    rʲ = cell_sol.u[j]
    xᵢʲ = rʲ[i]
    if i == firstindex(rʲ)
        qᵢ₊₁ʲ = cell_density(cell_sol, i + 1, j)
        qᵢ₊₂ʲ = cell_density(cell_sol, i + 2, j)
        xᵢ₊₁ʲ = rʲ[i+1]
        xᵢ₊₂ʲ = rʲ[i+2]
        ∂²q∂x² = forward_d²fdx²(qᵢʲ, qᵢ₊₁ʲ, qᵢ₊₂ʲ, xᵢʲ, xᵢ₊₁ʲ, xᵢ₊₂ʲ)
    elseif i == lastindex(rʲ)
        qᵢ₋₂ʲ = cell_density(cell_sol, i - 2, j)
        qᵢ₋₁ʲ = cell_density(cell_sol, i - 1, j)
        xᵢ₋₂ʲ = rʲ[i-2]
        xᵢ₋₁ʲ = rʲ[i-1]
        ∂²q∂x² = backward_d²fdx²(qᵢ₋₂ʲ, qᵢ₋₁ʲ, qᵢʲ, xᵢ₋₂ʲ, xᵢ₋₁ʲ, xᵢʲ)
    else
        qᵢ₋₁ʲ = cell_density(cell_sol, i - 1, j)
        qᵢ₊₁ʲ = cell_density(cell_sol, i + 1, j)
        xᵢ₋₁ʲ = rʲ[i-1]
        xᵢ₊₁ʲ = rʲ[i+1]
        ∂²q∂x² = central_d²fdx²(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, xᵢ₋₁ʲ, xᵢʲ, xᵢ₊₁ʲ)
    end
    return ∂²q∂x²
end

function cell_∂q∂t(cell_sol::ODESolution, i, j)
    if j == 1
        throw(ArgumentError("You should not include the initial condition in your matrix problem. Please redefine the time range."))
    end
    qᵢʲ = cell_density(cell_sol, i, j)
    qᵢʲ⁻¹, _ = cell_density(cell_sol.u[j-1], cell_sol.u[j][i], i)
    tʲ = cell_sol.t[j]
    tʲ⁻¹ = cell_sol.t[j-1]
    if j < lastindex(cell_sol)
        qᵢʲ⁺¹, _ = cell_density(cell_sol.u[j+1], cell_sol.u[j][i], i)
        tʲ⁺¹ = cell_sol.t[j+1]
        ∂q∂t = central_dfdx(qᵢʲ⁻¹, qᵢʲ, qᵢʲ⁺¹, tʲ⁻¹, tʲ, tʲ⁺¹)
    else # if j == lastindex(cell_sol)
        qᵢʲ⁻², _ = cell_density(cell_sol.u[j-2], cell_sol.u[j][i], i)
        tʲ⁻² = cell_sol.t[j-2]
        ∂q∂t = backward_dfdx(qᵢʲ⁻², qᵢʲ⁻¹, qᵢʲ, tʲ⁻², tʲ⁻¹, tʲ)
    end
    return ∂q∂t
end
function cell_∂q∂t(cell_sol::AveragedODESolution, i, j)
    if j == 1
        throw(ArgumentError("You should not include the initial condition in your matrix problem. Please redefine the time range."))
    end
    qᵢʲ = cell_density(cell_sol, i, j)
    qᵢʲ⁻¹ = LinearInterpolation{true}(cell_sol.q[j-1], cell_sol.u[j-1])(cell_sol.u[j][i])
    tʲ = cell_sol.t[j]
    tʲ⁻¹ = cell_sol.t[j-1]
    if j < lastindex(cell_sol)
        qᵢʲ⁺¹ = LinearInterpolation{true}(cell_sol.q[j+1], cell_sol.u[j+1])(cell_sol.u[j][i])
        tʲ⁺¹ = cell_sol.t[j+1]
        ∂q∂t = central_dfdx(qᵢʲ⁻¹, qᵢʲ, qᵢʲ⁺¹, tʲ⁻¹, tʲ, tʲ⁺¹)
    else # if j == lastindex(cell_sol)
        qᵢʲ⁻² = LinearInterpolation{true}(cell_sol.q[j-2], cell_sol.u[j-2])(cell_sol.u[j][i])
        tʲ⁻² = cell_sol.t[j-2]
        ∂q∂t = backward_dfdx(qᵢʲ⁻², qᵢʲ⁻¹, qᵢʲ, tʲ⁻², tʲ⁻¹, tʲ)
    end
    return ∂q∂t
end

function cell_dLdt(cell_sol, j)
    if j == 1
        throw(ArgumentError("You should not include the initial condition in your matrix problem. Please redefine the time range."))
    end
    Lⱼ = cell_sol.u[j][end]
    tⱼ = cell_sol.t[j]
    Lⱼ₋₁ = cell_sol.u[j-1][end]
    tⱼ₋₁ = cell_sol.t[j-1]
    if j < lastindex(cell_sol)
        Lⱼ₊₁ = cell_sol.u[j+1][end]
        tⱼ₊₁ = cell_sol.t[j+1]
        dLdt = central_dfdx(Lⱼ₋₁, Lⱼ, Lⱼ₊₁, tⱼ₋₁, tⱼ, tⱼ₊₁)
    else # if j == lastindex(cell_sol)
        Lⱼ₋₂ = cell_sol.u[j-2][end]
        tⱼ₋₂ = cell_sol.t[j-2]
        dLdt = backward_dfdx(Lⱼ₋₂, Lⱼ₋₁, Lⱼ, tⱼ₋₂, tⱼ₋₁, tⱼ)
    end
    return dLdt
end