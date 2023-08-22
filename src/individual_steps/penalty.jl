@inline function parsimony_penalty(indicators, k=2)
    return k * count(indicators)
end

function check_negative_diffusion_or_moving_boundary(model, θ, num_constraint_checks, conserve_mass)
    qmin = model.qmin
    qmax = model.qmax
    Δq = (qmax - qmin) / (num_constraint_checks - 1)
    nd = num_diffusion(model)
    if nd ≠ 0
        p = model.diffusion_parameters
        D = model.diffusion_basis
        θd = @views θ[1:nd]
        diffusion_flag = any(k -> D(qmin + (k - 1) * Δq, θd, p) < 0.0, 1:num_constraint_checks)
        diffusion_flag && return true
    end
    nm = num_moving_boundary(model)
    if nm ≠ 0 && !conserve_mass
        p = model.moving_boundary_parameters
        E = model.moving_boundary_basis
        θm = @views θ[(nd+num_reaction(model)+num_rhs(model)+1):(nd+num_reaction(model)+num_rhs(model)+nm)]
        moving_boundary_flag = any(k -> E(qmin + (k - 1) * Δq, θm, p) < 0.0, 1:num_constraint_checks)
        moving_boundary_flag && return true
    end
    return false
end