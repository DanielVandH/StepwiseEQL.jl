struct EQLModel{S,P,D,Dp,Dθ,R,Rp,Rθ,RH,RHp,RHθ,MB,MBp,MBθ,SI}
    A::Matrix{Float64}
    b::Vector{Float64}
    idx_map::Bijection{Int,NTuple{3,Int}} # row ↦ (k, i, j), k = parent_sim, i = cell, j = time
    valid_time_indices::Vector{Int}
    cell_sol::S
    pde_template::P
    diffusion_basis::D
    diffusion_parameters::Dp
    diffusion_theta::Dθ
    reaction_basis::R
    reaction_parameters::Rp
    reaction_theta::Rθ
    rhs_basis::RH
    rhs_parameters::RHp
    rhs_theta::RHθ
    moving_boundary_basis::MB
    moving_boundary_parameters::MBp
    moving_boundary_theta::MBθ
    indicators::Vector{Bool} # BitVector is not thread-safe. Just easier to avoid ever having to deal with it
    simulation_indices::SI
    qmin::Float64
    qmax::Float64
end

function Base.show(io::IO, ::MIME"text/plain", model::EQLModel)
    neqs = size(model.A, 1)
    ndiff = isnothing(model.diffusion_basis) ? 0 : length(model.diffusion_basis)
    nreact = isnothing(model.reaction_basis) ? 0 : length(model.reaction_basis)
    nrhs = isnothing(model.rhs_basis) ? 0 : length(model.rhs_basis)
    nmb = isnothing(model.moving_boundary_basis) ? 0 : length(model.moving_boundary_basis)
    println(io, "EQLModel.")
    println(io, "    Num equations: ", neqs)
    println(io, "    Diffusion library length: ", ndiff)
    println(io, "    Reaction library length: ", nreact)
    println(io, "    RHS library length: ", nrhs)
    println(io, "    Moving boundary library length: ", nmb)
    if model.cell_sol isa EnsembleSolution
        print(io, "    Time range: (", model.cell_sol[begin].t[model.valid_time_indices[begin]], ", ", model.cell_sol[begin].t[model.valid_time_indices[end]], ")")
    else
        print(io, "    Time range: (", model.cell_sol.t[model.valid_time_indices[begin]], ", ", model.cell_sol.t[model.valid_time_indices[end]], ")")
    end
end

function EQLModel(cell_sol;
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
    threshold_tol::NamedTuple=(q=0.0, x=0.0, dt=0.0, dx=0.0, dx2=0.0, dx_bc=0.0, dL=0.0),
    time_range=default_time_range(cell_sol),
    mesh_points=1000,
    pde=build_pde(cell_sol,
        mesh_points;
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
        moving_boundary_theta),
    average=Val(true),
    simulation_indices=eachindex(cell_sol),
    num_knots=100,
    avg_interp_fnc=LinearInterpolation{true},
    stat=mean,
    conserve_mass=false,
    extrapolate_average=true,
    couple_rhs=false)
    if conserve_mass
        moving_boundary_basis = diffusion_basis
        moving_boundary_parameters = diffusion_parameters
        moving_boundary_theta = diffusion_theta
    end
    A, b, idx_map, valid_time_indices, qmin, qmax, _sol = build_basis_system(
        cell_sol;
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
        threshold_tol,
        time_range,
        average,
        simulation_indices,
        num_knots,
        avg_interp_fnc,
        stat,
        conserve_mass,
        extrapolate_average,
        couple_rhs)
    return EQLModel(
        A,
        b,
        idx_map,
        valid_time_indices,
        _sol,
        pde,
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
        fill(true, size(A, 2)),
        simulation_indices,
        qmin,
        qmax
    )
end
num_diffusion(model::EQLModel) = _length(model.diffusion_basis, model.diffusion_theta)
num_reaction(model::EQLModel) = _length(model.reaction_basis, model.reaction_theta)
num_rhs(model::EQLModel) = _length(model.rhs_basis, model.rhs_theta)
num_moving_boundary(model::EQLModel) = _length(model.moving_boundary_basis, model.moving_boundary_theta)
is_diffusion_index(model, i) = i ≤ num_diffusion(model)
is_reaction_index(model, i) = num_diffusion(model) < i ≤ num_diffusion(model) + num_reaction(model)
is_rhs_index(model, i) = num_diffusion(model) + num_reaction(model) < i ≤ num_diffusion(model) + num_reaction(model) + num_rhs(model)
is_moving_boundary_index(model, i) = num_diffusion(model) + num_reaction(model) + num_rhs(model) < i ≤ num_diffusion(model) + num_reaction(model) + num_rhs(model) + num_moving_boundary(model)
