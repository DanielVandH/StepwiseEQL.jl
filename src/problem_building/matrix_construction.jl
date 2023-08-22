_num_eqs(cell_sol::Union{ODESolution,AveragedODESolution}) = sum(length(u) for u in cell_sol.u)
_num_eqs(cell_sol::EnsembleSolution) = sum(_num_eqs(sol) for sol in cell_sol)
_length(basis, theta) = length(basis) * isnothing(theta)
is_fixed(basis, theta) = _length(basis, theta) == 0
@inline function get_pde_terms(cell_sol, i, j)
    q = cell_density(cell_sol, i, j)
    ∂q∂x = cell_∂q∂x(cell_sol, i, j)
    ∂q∂t = cell_∂q∂t(cell_sol, i, j)
    ∂²q∂x² = cell_∂²q∂x²(cell_sol, i, j)
    return q, ∂q∂x, ∂q∂t, ∂²q∂x²
end
@inline function passes_filter(q, ∂q∂t, ∂q∂x, ∂²q∂x², x, qmin, qmax, ∂ₜqmin, ∂ₜqmax, ∂ₓqmin, ∂ₓqmax, ∂²ₓqmin, ∂²ₓqmax, L, xtol)
    cond = (qmin ≤ q ≤ qmax) &&
           (∂ₜqmin ≤ abs(∂q∂t) ≤ ∂ₜqmax) &&
           (∂ₓqmin ≤ abs(∂q∂x) ≤ ∂ₓqmax) &&
           (∂²ₓqmin ≤ abs(∂²q∂x²) ≤ ∂²ₓqmax) &&
           (0 ≤ x ≤ (1 - xtol) * L)
    return cond
end
@inline function passes_filter(∂q∂x_bc, ∂q∂xmin, ∂q∂xmax) # also used for dLdt
    cond = ∂q∂xmin ≤ abs(∂q∂x_bc) ≤ ∂q∂xmax
    return cond
end

default_time_range(cell_sol::ODESolution) = (cell_sol.t[begin+1], cell_sol.t[end])
default_time_range(cell_sol::EnsembleSolution) = default_time_range(cell_sol[begin])

in_time_range(cell_sol, j, time_range) = time_range[1] ≤ cell_sol.t[j] ≤ time_range[2]
function get_valid_time_indices(cell_sol::Union{ODESolution,AveragedODESolution}, time_range)
    valid_time_indices = Int[]
    for j in eachindex(cell_sol)
        in_time_range(cell_sol, j, time_range) && push!(valid_time_indices, j)
    end
    return valid_time_indices
end
get_valid_time_indices(cell_sol::EnsembleSolution, time_range) = get_valid_time_indices(cell_sol[begin], time_range)

function get_vecs_for_quantiles(cell_sol::Union{ODESolution,AveragedODESolution}, valid_time_indices, simulation_indices=nothing) # last kwarg is ignored, it's just here for the other method with an EnsembleSolution
    q = Float64[]
    ∂q∂t = Float64[]
    ∂q∂x = Float64[]
    ∂²q∂x² = Float64[]
    ∂q∂x_bc = Float64[]
    dLdt = Float64[]
    for j in valid_time_indices
        for i in eachindex(cell_sol.u[j])
            push!(q, (abs ∘ cell_density)(cell_sol, i, j))
            push!(∂q∂t, (abs ∘ cell_∂q∂t)(cell_sol, i, j))
            push!(∂q∂x, (abs ∘ cell_∂q∂x)(cell_sol, i, j))
            push!(∂²q∂x², (abs ∘ cell_∂²q∂x²)(cell_sol, i, j))
        end
        push!(∂q∂x_bc, (abs ∘ cell_∂q∂x)(cell_sol, lastindex(cell_sol.u[j]), j))
        push!(dLdt, (abs ∘ cell_dLdt)(cell_sol, j))
    end
    return q, ∂q∂t, ∂q∂x, ∂²q∂x², ∂q∂x_bc, dLdt
end
function get_vecs_for_quantiles(cell_sol::EnsembleSolution, valid_time_indices, simulation_indices)
    q = Float64[]
    ∂q∂t = Float64[]
    ∂q∂x = Float64[]
    ∂²q∂x² = Float64[]
    ∂q∂x_bc = Float64[]
    dLdt = Float64[]
    for k in simulation_indices
        sol = cell_sol[k]
        _q, _∂q∂t, _∂q∂x, _∂²q∂x², _∂q∂x_bc, _dLdt = get_vecs_for_quantiles(sol, valid_time_indices)
        append!(q, _q)
        append!(∂q∂t, _∂q∂t)
        append!(∂q∂x, _∂q∂x)
        append!(∂²q∂x², _∂²q∂x²)
        append!(∂q∂x_bc, _∂q∂x_bc)
        append!(dLdt, _dLdt)
    end
    return q, ∂q∂t, ∂q∂x, ∂²q∂x², ∂q∂x_bc, dLdt
end

@memoize function get_extrema_vals(cell_sol, valid_time_indices, threshold_tol, simulation_indices)
    def_tol = (q=0.0, x=0.0, dt=0.0, dx=0.0, dx2=0.0, dx_bc=0.0, dL=0.0)
    threshold_tol = merge(def_tol, threshold_tol)
    @assert all(((f, x),) -> f ≠ :x ? (0 ≤ x < 1 / 2) : (0 ≤ x ≤ 1), pairs(threshold_tol)) "All threshold tolerances (except for x, which is in [0, 1]) must be in [0, 1/2)."
    if length(threshold_tol) ≠ 7
        throw("threshold_tol must be a NamedTuple with 7 fields: q, x, dt, dx, dx2, dx_bc, dL")
    end
    q, ∂q∂t, ∂q∂x, ∂²q∂x², ∂q∂x_bc, dLdt = get_vecs_for_quantiles(cell_sol, valid_time_indices, simulation_indices)
    τq = threshold_tol.q
    τdt = threshold_tol.dt
    τdx = threshold_tol.dx
    τdx2 = threshold_tol.dx2
    τdx_bc = threshold_tol.dx_bc
    τdL = threshold_tol.dL
    qmin, qmax = quantile(q, (τq, 1 - τq))
    ∂ₜqmin, ∂ₜqmax = quantile(∂q∂t, (τdt, 1 - τdt))
    ∂ₓqmin, ∂ₓqmax = quantile(∂q∂x, (τdx, 1 - τdx))
    ∂²ₓqmin, ∂²ₓqmax = quantile(∂²q∂x², (τdx2, 1 - τdx2))
    ∂ₓqmin_bc, ∂ₓqmax_bc = quantile(∂q∂x_bc, (τdx_bc, 1 - τdx_bc))
    dLdtmin, dLdtmax = quantile(dLdt, (τdL, 1 - τdL))
    return qmin, qmax, ∂ₜqmin, ∂ₜqmax, ∂ₓqmin, ∂ₓqmax, ∂²ₓqmin, ∂²ₓqmax, ∂ₓqmin_bc, ∂ₓqmax_bc, dLdtmin, dLdtmax, threshold_tol, minimum(q)::Float64, maximum(q)::Float64
end

function prepare_basis_system_arrays(cell_sol,
    diffusion_basis, diffusion_theta,
    reaction_basis, reaction_theta,
    rhs_basis, rhs_theta,
    moving_boundary_basis, moving_boundary_theta,
    threshold_tol, time_range, average,
    simulation_indices, num_knots, avg_interp_fnc,
    stat, conserve_mass, extrapolate_average)
    if average === Val(true) && cell_sol isa EnsembleSolution
        _cell_sol = AveragedODESolution(cell_sol, num_knots, simulation_indices, avg_interp_fnc, stat, extrapolate_average)
    else
        _cell_sol = cell_sol
    end
    nᵈ = _length(diffusion_basis, diffusion_theta)
    nʳ = _length(reaction_basis, reaction_theta)
    nʰ = _length(rhs_basis, rhs_theta)
    nᵉ = !conserve_mass ? _length(moving_boundary_basis, moving_boundary_theta) : 0
    A1 = ElasticMatrix{Float64}(undef, nᵈ + nʳ, 0)
    A2 = ElasticMatrix{Float64}(undef, nʰ, 0)
    A3 = ElasticMatrix{Float64}(undef, nᵉ, 0)
    b1 = Float64[]
    b2 = Float64[]
    b3 = Float64[]
    idx_map = Bijection{Int,NTuple{3,Int}}() # row ↦ (k, i, j), k = parent_sim, i = cell, j = time
    valid_time_indices = get_valid_time_indices(_cell_sol, time_range)
    qmin, qmax, ∂ₜqmin, ∂ₜqmax, ∂ₓqmin, ∂ₓqmax, ∂²ₓqmin, ∂²ₓqmax, ∂ₓqmin_bc, ∂ₓqmax_bc, dLdtmin, dLdtmax, threshold_tol, _qmin, _qmax = get_extrema_vals(_cell_sol, valid_time_indices, threshold_tol, simulation_indices)
    num_eqs = _num_eqs(_cell_sol)
    nᵈ + nʳ + nʰ + nᵉ == 0 && throw("Cannot leave all mechanisms fixed.")
    nᵈ + nʳ > 0 && sizehint!(A1, (nᵈ + nʳ, num_eqs))
    sizehint!(b1, num_eqs)
    nʰ > 0 && sizehint!(A2, (nʰ, length(valid_time_indices)))
    sizehint!(b2, length(valid_time_indices))
    nᵉ > 0 && sizehint!(A3, (nᵉ, length(valid_time_indices)))
    sizehint!(b3, length(valid_time_indices))
    sizehint!(idx_map, num_eqs)
    return A1, A2, A3, b1, b2, b3, idx_map, valid_time_indices, qmin, qmax, ∂ₜqmin, ∂ₜqmax, ∂ₓqmin, ∂ₓqmax, ∂²ₓqmin, ∂²ₓqmax, ∂ₓqmin_bc, ∂ₓqmax_bc, dLdtmin, dLdtmax, num_eqs, _cell_sol, threshold_tol, _qmin, _qmax
end

function build_basis_system!(A1, A2, A3, b1, b2, b3, idx_map, valid_time_indices, cell_sol::Union{ODESolution,AveragedODESolution}, parent_sim=1;
    diffusion_basis,
    diffusion_parameters=nothing,
    diffusion_theta=nothing,
    reaction_basis=BasisSet(),
    reaction_parameters=nothing,
    reaction_theta=nothing,
    rhs_basis=BasisSet(),
    rhs_parameters=nothing,
    rhs_theta=nothing,
    moving_boundary_basis=BasisSet(),
    moving_boundary_parameters=nothing,
    moving_boundary_theta=nothing,
    qmin=0.0,
    qmax=Inf,
    ∂ₜqmin=0.0,
    ∂ₜqmax=Inf,
    ∂ₓqmin=0.0,
    ∂ₓqmax=Inf,
    ∂²ₓqmin=0.0,
    ∂²ₓqmax=Inf,
    ∂ₓqmin_bc=0.0,
    ∂ₓqmax_bc=Inf,
    dLdtmin=0.0,
    dLdtmax=Inf,
    xtol=0.0,
    num_eqs=_num_eqs(cell_sol),
    conserve_mass=false,
    couple_rhs=false,
    simulation_indices=nothing) # the last kwarg is ignored, it's just here for the other method with an EnsembleSolution   
    N1 = size(A1, 1)
    N2 = size(A2, 1)
    N3 = size(A3, 1)
    nᵈ = _length(diffusion_basis, diffusion_theta)
    nʳ = _length(reaction_basis, reaction_theta)
    nʰ = _length(rhs_basis, rhs_theta)
    nᵉ = !conserve_mass ? _length(moving_boundary_basis, moving_boundary_theta) : 0
    for j in valid_time_indices
        # Get A1 
        if N1 > 0
            for i in eachindex(cell_sol.u[j])
                q, ∂q∂x, ∂q∂t, ∂²q∂x² = get_pde_terms(cell_sol, i, j)
                if passes_filter(q, ∂q∂t, ∂q∂x, ∂²q∂x², cell_sol.u[j][i], qmin, qmax, ∂ₜqmin, ∂ₜqmax, ∂ₓqmin, ∂ₓqmax, ∂²ₓqmin, ∂²ₓqmax, cell_sol.u[j][end], xtol)
                    append!(A1, ntuple(i -> 0.0, N1))
                    idx_map[size(A1, 2)] = (parent_sim, i, j)
                    for k in 1:nᵈ
                        ϕₖ, ϕₖ′ = eval_and_diff(diffusion_basis, q, diffusion_parameters, k)
                        A1[k, end] = ϕₖ′ * ∂q∂x^2 + ϕₖ * ∂²q∂x²
                    end
                    for k in (nᵈ+1):(nᵈ+nʳ)
                        A1[k, end] = reaction_basis.bases[k-nᵈ](q, reaction_parameters)
                    end
                    bᵢⱼ = ∂q∂t
                    if is_fixed(diffusion_basis, diffusion_theta) && length(diffusion_basis) > 0
                        D, D′ = eval_and_diff(diffusion_basis, q, diffusion_theta, diffusion_parameters)
                        bᵢⱼ -= (D′ * ∂q∂x^2 + D * ∂²q∂x²)
                    end
                    if is_fixed(reaction_basis, reaction_theta) && length(reaction_basis) > 0
                        R = reaction_basis(q, reaction_theta, reaction_parameters)
                        bᵢⱼ -= R
                    end
                    push!(b1, bᵢⱼ)
                end
            end
        end

        # Get A2
        q, ∂q∂x, ∂q∂t, ∂²q∂x² = get_pde_terms(cell_sol, lastindex(cell_sol.u[j]), j)
        dLdt = cell_dLdt(cell_sol, j)
        if N2 > 0 && passes_filter(∂q∂x, ∂ₓqmin_bc, ∂ₓqmax_bc) #=&& passes_filter(dLdt, dLdtmin, dLdtmax)=#
            append!(A2, ntuple(i -> 0.0, N2))
            idx_map[5num_eqs+size(A2, 2)] = (parent_sim, 0, j)
            for k in 1:nʰ
                A2[k, end] = rhs_basis.bases[k](q, rhs_parameters)
            end
            push!(b2, ∂q∂x)
            if is_fixed(moving_boundary_basis, moving_boundary_theta) && length(moving_boundary_basis) > 0 && couple_rhs && passes_filter(dLdt, dLdtmin, dLdtmax)
                append!(A2, ntuple(i -> 0.0, N2))
                idx_map[5num_eqs+size(A2, 2)] = (parent_sim, -1, j)
                E = moving_boundary_basis(q, moving_boundary_theta, moving_boundary_parameters)
                for k in 1:nʰ
                    A2[k, end] = -E * rhs_basis.bases[k](q, rhs_parameters)
                end
                push!(b2, q * dLdt)
            end
        end

        if !conserve_mass
            # Get A3 
            if N3 > 0 && passes_filter(dLdt, dLdtmin, dLdtmax) #&& passes_filter(∂q∂x, ∂ₓqmin_bc, ∂ₓqmax_bc)
                append!(A3, ntuple(i -> 0.0, N3))
                idx_map[10num_eqs+length(valid_time_indices)+1+size(A3, 2)] = (parent_sim, -1, j)
                for k in 1:nᵉ
                    A3[k, end] = -∂q∂x * moving_boundary_basis.bases[k](q, moving_boundary_parameters)
                end
                push!(b3, q * dLdt)
            end
        else
            # Add to A1 if we have to conserve mass 
            if N1 > 0 && passes_filter(dLdt, dLdtmin, dLdtmax) #&& passes_filter(∂q∂x, ∂ₓqmin_bc, ∂ₓqmax_bc)
                append!(A1, ntuple(i -> 0.0, N1))
                idx_map[size(A1, 2)] = (parent_sim, -1, j)
                for k in 1:nᵈ
                    A1[k, end] = -∂q∂x * diffusion_basis.bases[k](q, diffusion_parameters)
                end
                push!(b1, q * dLdt)
            end
        end
    end
    return nothing
end

function build_basis_system(
    cell_sol;
    diffusion_basis,
    diffusion_parameters=nothing,
    diffusion_theta=nothing,
    reaction_basis=BasisSet(),
    reaction_parameters=nothing,
    reaction_theta=nothing,
    rhs_basis=BasisSet(),
    rhs_parameters=nothing,
    rhs_theta=nothing,
    moving_boundary_basis=BasisSet(),
    moving_boundary_parameters=nothing,
    moving_boundary_theta=nothing,
    threshold_tol=(q=0.0, x=0.0, dt=0.0, dx=0.0, dx2=0.0, dx_bc=0.0, dL=0.0),
    time_range=default_time_range(cell_sol),
    average=Val(true),
    simulation_indices=eachindex(cell_sol),
    num_knots=100,
    avg_interp_fnc=LinearInterpolation{true},
    stat=mean,
    conserve_mass=false,
    extrapolate_average=true,
    couple_rhs=false
)
    @assert average isa Val
    A1, A2, A3,
    b1, b2, b3,
    idx_map, valid_time_indices,
    qmin, qmax,
    ∂ₜqmin, ∂ₜqmax,
    ∂ₓqmin, ∂ₓqmax,
    ∂²ₓqmin, ∂²ₓqmax,
    ∂ₓqmin_bc, ∂ₓqmax_bc,
    dLdtmin, dLdtmax, num_eqs,
    _cell_sol, threshold_tol,
    _qmin, _qmax = prepare_basis_system_arrays(cell_sol,
        diffusion_basis, diffusion_theta,
        reaction_basis, reaction_theta,
        rhs_basis, rhs_theta,
        moving_boundary_basis, moving_boundary_theta,
        threshold_tol, time_range, average, simulation_indices,
        num_knots, avg_interp_fnc, stat, conserve_mass,
        extrapolate_average)
    build_basis_system!(A1, A2, A3, b1, b2, b3, idx_map, valid_time_indices, _cell_sol;
        diffusion_basis,
        diffusion_parameters,
        diffusion_theta,
        reaction_basis,
        reaction_parameters,
        reaction_theta,
        rhs_basis,
        rhs_parameters,
        rhs_theta,
        moving_boundary_basis,
        moving_boundary_parameters,
        moving_boundary_theta,
        qmin,
        qmax,
        ∂ₜqmin,
        ∂ₜqmax,
        ∂ₓqmin,
        ∂ₓqmax,
        ∂²ₓqmin,
        ∂²ₓqmax,
        ∂ₓqmin_bc,
        ∂ₓqmax_bc,
        dLdtmin,
        dLdtmax,
        num_eqs,
        xtol=threshold_tol.x,
        simulation_indices,
        couple_rhs,
        conserve_mass)
    new_idx_map = fix_idx_map(idx_map, A1, A2, A3, num_eqs, valid_time_indices)
    A1 = A1'
    A2 = A2'
    A3 = A3'
    A = BlockDiagonal([A1, A2, A3])
    b = vcat(b1, b2, b3)
    return Matrix(A), b, new_idx_map, valid_time_indices, _qmin::Float64, _qmax::Float64, _cell_sol
end

function fix_idx_map(idx_map, A1, A2, A3, num_eqs, valid_time_indices) # make the idx keys contiguous 
    new_idx_map = Bijection{Int,NTuple{3,Int}}() # row ↦ (k, i, j), k = parent_sim, i = cell, j = time
    # ctr 
    # 5num_eqs + ctr 
    # 10num_eqs + length(valid_time_indices) + 1 + ctr
    for i in axes(A1, 2)
        new_idx_map[i] = idx_map[i]
    end
    for i in axes(A2, 2)
        new_idx_map[size(A1, 2)+i] = idx_map[5num_eqs+i]
    end
    for i in axes(A3, 2)
        new_idx_map[size(A1, 2)+size(A2, 2)+i] = idx_map[10num_eqs+length(valid_time_indices)+1+i]
    end
    return new_idx_map
end

function build_basis_system!(A1, A2, A3, b1, b2, b3, idx_map, valid_time_indices, cell_sol::EnsembleSolution;
    diffusion_basis,
    diffusion_parameters=nothing,
    diffusion_theta=nothing,
    reaction_basis=BasisSet(),
    reaction_parameters=nothing,
    reaction_theta=nothing,
    rhs_basis=BasisSet(),
    rhs_parameters=nothing,
    rhs_theta=nothing,
    moving_boundary_basis=BasisSet(),
    moving_boundary_parameters=nothing,
    moving_boundary_theta=nothing,
    qmin=0.0,
    qmax=Inf,
    ∂ₜqmin=0.0,
    ∂ₜqmax=Inf,
    ∂ₓqmin=0.0,
    ∂ₓqmax=Inf,
    ∂²ₓqmin=0.0,
    ∂²ₓqmax=Inf,
    ∂ₓqmin_bc=0.0,
    ∂ₓqmax_bc=Inf,
    dLdtmin=0.0,
    dLdtmax=Inf,
    xtol=0.0,
    num_eqs=_num_eqs(cell_sol),
    simulation_indices=eachindex(cell_sol),
    conserve_mass=false,
    couple_rhs=false)
    @assert allunique(simulation_indices) "When considering an aggregated EnsembleSolution, simulation_indices must be unique."
    for k in simulation_indices
        sol = cell_sol[k]
        build_basis_system!(A1, A2, A3, b1, b2, b3, idx_map, valid_time_indices, sol, k;
            diffusion_basis,
            diffusion_parameters,
            diffusion_theta,
            reaction_basis,
            reaction_parameters,
            reaction_theta,
            rhs_basis,
            rhs_parameters,
            rhs_theta,
            moving_boundary_basis,
            moving_boundary_parameters,
            moving_boundary_theta,
            qmin,
            qmax,
            ∂ₜqmin,
            ∂ₜqmax,
            ∂ₓqmin,
            ∂ₓqmax,
            ∂²ₓqmin,
            ∂²ₓqmax,
            ∂ₓqmin_bc,
            ∂ₓqmax_bc,
            dLdtmin,
            dLdtmax,
            xtol,
            num_eqs,
            conserve_mass,
            couple_rhs)
    end
    return nothing
end