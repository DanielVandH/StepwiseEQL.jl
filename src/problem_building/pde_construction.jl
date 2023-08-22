has_proliferation(cell_sol::ODESolution) = length(cell_sol[begin]) ≠ length(cell_sol[end])
has_proliferation(cell_sol::EnsembleSolution) = any(has_proliferation, cell_sol.u)
has_proliferation(cell_sol::AveragedODESolution) = has_proliferation(cell_sol.cell_sol)
has_moving_boundary(cell_sol::ODESolution) = !cell_sol.prob.p.fix_right
has_moving_boundary(cell_sol::EnsembleSolution) = any(has_moving_boundary, cell_sol.u)
has_moving_boundary(cell_sol::AveragedODESolution) = has_moving_boundary(cell_sol.cell_sol)
get_saveat(cell_sol::Union{ODESolution,AveragedODESolution}) = cell_sol.t
get_saveat(cell_sol::EnsembleSolution) = get_saveat(cell_sol[begin])
get_problem(cell_sol::ODESolution) = cell_sol.prob.p
get_problem(cell_sol::EnsembleSolution) = get_problem(cell_sol[begin])
get_problem(cell_sol::AveragedODESolution) = get_problem(cell_sol.cell_sol)
function build_pde(cell_sol, mesh_points=1000;
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
    conserve_mass=false)
    if !isnothing(diffusion_theta) && conserve_mass
        throw("Cannot conserve mass and fix D(q).")
    end
    if isnothing(diffusion_theta)
        diffusion_theta = zeros(length(diffusion_basis))
    end
    if isnothing(reaction_theta)
        reaction_theta = zeros(length(reaction_basis))
    end
    if isnothing(rhs_theta)
        rhs_theta = zeros(length(rhs_basis))
    end
    if isnothing(moving_boundary_theta)
        moving_boundary_theta = zeros(length(moving_boundary_basis))
    end
    proliferation = has_proliferation(cell_sol)
    moving = has_moving_boundary(cell_sol)
    if proliferation
        if !moving
            return _build_fvm_pde_proliferation(cell_sol, mesh_points;
                diffusion_basis,
                reaction_basis,
                diffusion_parameters,
                reaction_parameters,
                diffusion_theta,
                reaction_theta)
        else
            return _build_mb_pbe_proliferation(cell_sol, mesh_points;
                diffusion_basis,
                reaction_basis,
                rhs_basis,
                moving_boundary_basis,
                diffusion_parameters,
                reaction_parameters,
                rhs_parameters,
                moving_boundary_parameters,
                diffusion_theta,
                reaction_theta,
                rhs_theta,
                moving_boundary_theta,
                conserve_mass)
        end
    else
        if !moving
            return _build_fvm_pde_no_proliferation(cell_sol, mesh_points;
                diffusion_basis,
                diffusion_parameters,
                diffusion_theta)
        else
            return _build_mb_pbe_no_proliferation(cell_sol, mesh_points;
                diffusion_basis,
                rhs_basis,
                moving_boundary_basis,
                diffusion_parameters,
                rhs_parameters,
                moving_boundary_parameters,
                diffusion_theta,
                rhs_theta,
                moving_boundary_theta,
                conserve_mass)
        end
    end
    return pde_prob
end

function _build_fvm_pde_no_proliferation(cell_sol, mesh_points=1000;
    diffusion_basis,
    diffusion_parameters=nothing,
    diffusion_theta=zeros(length(diffusion_basis)))
    prob = get_problem(cell_sol)
    pde_prob = FVMProblem(prob, mesh_points;
        diffusion_function=diffusion_basis,
        diffusion_parameters=Parameters(θ=diffusion_theta, p=diffusion_parameters),
        proliferation=false
    )
    return pde_prob
end
function _build_fvm_pde_proliferation(cell_sol, mesh_points=1000;
    diffusion_basis,
    reaction_basis,
    diffusion_parameters=nothing,
    reaction_parameters=nothing,
    diffusion_theta=zeros(length(diffusion_basis)),
    reaction_theta=zeros(length(reaction_basis)))
    prob = get_problem(cell_sol)
    pde_prob = FVMProblem(prob, mesh_points;
        diffusion_function=diffusion_basis,
        diffusion_parameters=Parameters(θ=diffusion_theta, p=diffusion_parameters),
        reaction_function=reaction_basis,
        reaction_parameters=Parameters(θ=reaction_theta, p=reaction_parameters),
        proliferation=true
    )
    return pde_prob
end

function _build_mb_pbe_no_proliferation(cell_sol, mesh_points=1000;
    diffusion_basis,
    rhs_basis,
    moving_boundary_basis,
    diffusion_parameters=nothing,
    rhs_parameters=nothing,
    moving_boundary_parameters=nothing,
    diffusion_theta=zeros(length(diffusion_basis)),
    rhs_theta=zeros(length(rhs_basis)),
    moving_boundary_theta=zeros(length(moving_boundary_basis)),
    conserve_mass=false)
    prob = get_problem(cell_sol)
    pde_prob = MBProblem(prob, mesh_points;
        diffusion_function=diffusion_basis,
        diffusion_parameters=Parameters(θ=diffusion_theta, p=diffusion_parameters),
        rhs_function=rhs_basis,
        rhs_parameters=Parameters(θ=rhs_theta, p=rhs_parameters),
        moving_boundary_function=conserve_mass ? (u, t, p) -> (zero(u), -inv(u) * moving_boundary_basis(u, t, p)) : (u, t, p) -> (zero(u), -inv(u) * diffusion_basis(u, t, p)),
        moving_boundary_parameters=conserve_mass ? Parameters(θ=diffusion_theta, p=diffusion_parameters) : Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters),
        proliferation=false)
    return pde_prob
end
function _build_mb_pbe_proliferation(cell_sol, mesh_points=1000;
    diffusion_basis,
    reaction_basis,
    rhs_basis,
    moving_boundary_basis,
    diffusion_parameters=nothing,
    reaction_parameters=nothing,
    rhs_parameters=nothing,
    moving_boundary_parameters=nothing,
    diffusion_theta=zeros(length(diffusion_basis)),
    reaction_theta=zeros(length(reaction_basis)),
    rhs_theta=zeros(length(rhs_basis)),
    moving_boundary_theta=zeros(length(moving_boundary_basis)),
    conserve_mass=false)
    prob = get_problem(cell_sol)
    pde_prob = MBProblem(prob, mesh_points;
        diffusion_function=diffusion_basis,
        diffusion_parameters=Parameters(θ=diffusion_theta, p=diffusion_parameters),
        reaction_function=reaction_basis,
        reaction_parameters=Parameters(θ=reaction_theta, p=reaction_parameters),
        rhs_function=rhs_basis,
        rhs_parameters=Parameters(θ=rhs_theta, p=rhs_parameters),
        moving_boundary_function=conserve_mass ? (u, t, p) -> (zero(u), -inv(u) * moving_boundary_basis(u, t, p)) : (u, t, p) -> (zero(u), -inv(u) * diffusion_basis(u, t, p)),
        moving_boundary_parameters=conserve_mass ? Parameters(θ=diffusion_theta, p=diffusion_parameters) : Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters),
        proliferation=true)
    return pde_prob
end

function rebuild_pde(model)
    pde = model.pde_template
    if pde.diffusion_parameters isa Parameters && all(iszero, pde.diffusion_parameters.θ)
        pde = @set pde.diffusion_parameters.θ = zeros(length(pde.diffusion_parameters.θ))
    end
    if pde.reaction_parameters isa Parameters && all(iszero, pde.reaction_parameters.θ)
        pde = @set pde.reaction_parameters.θ = zeros(length(pde.reaction_parameters.θ))
    end
    if pde isa MBProblem
        if pde.boundary_conditions.rhs.p isa Parameters && all(iszero, pde.boundary_conditions.rhs.p.θ)
            pde = @set pde.boundary_conditions.rhs.p.θ = zeros(length(pde.boundary_conditions.rhs.p.θ))
        end
        if pde.boundary_conditions.moving_boundary.p isa Parameters && all(iszero, pde.boundary_conditions.moving_boundary.p.θ)
            pde = @set pde.boundary_conditions.moving_boundary.p.θ = zeros(length(pde.boundary_conditions.moving_boundary.p.θ))
        end
    end
    return pde
end
function rebuild_pde(model, θ_train, indicators=model.indicators, conserve_mass=false)
    pde = rebuild_pde(model)
    diffusion_free = pde.diffusion_parameters isa Parameters && all(iszero, pde.diffusion_parameters.θ)
    reaction_free = pde.reaction_parameters isa Parameters && all(iszero, pde.reaction_parameters.θ)
    rhs_free = !(pde isa MBProblem) || (pde.boundary_conditions.rhs.p isa Parameters && all(iszero, pde.boundary_conditions.rhs.p.θ))
    moving_boundary_free = !(pde isa MBProblem) || (pde.boundary_conditions.moving_boundary.p isa Parameters && all(iszero, pde.boundary_conditions.moving_boundary.p.θ))
    for (i, flag) in enumerate(indicators)
        !flag && continue
        if diffusion_free && is_diffusion_index(model, i)
            pde.diffusion_parameters.θ[i] = θ_train[i]
            if pde isa MBProblem && conserve_mass
                pde.boundary_conditions.moving_boundary.p.θ[i] = θ_train[i]
            end
        elseif reaction_free && is_reaction_index(model, i)
            pde.reaction_parameters.θ[i-num_diffusion(model)] = θ_train[i]
        elseif rhs_free && is_rhs_index(model, i)
            pde.boundary_conditions.rhs.p.θ[i-num_diffusion(model)-num_reaction(model)] = θ_train[i]
        elseif moving_boundary_free && is_moving_boundary_index(model, i) && !conserve_mass
            pde.boundary_conditions.moving_boundary.p.θ[i-num_diffusion(model)-num_reaction(model)-num_rhs(model)] = θ_train[i]
        end
    end
    return pde
end