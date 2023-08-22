"""
    stepwise_selection(cell_sol; kwargs...)

Perform stepwise selection on the given cell solution `cell_sol`, given as a 
`CellSolution` from EpithelialDynamics1D.jl. In the 
list of keyword arguments below, the notation `(A/B/C/D)_Y`, for example, 
means that there are keyword arguments `A_Y`, `B_Y`, `C_Y`, and `D_Y`.

The result returned from this function will be one of:

- `EQLSolution`

This gives the solution. You can display this in the REPL to see the table of 
results automatically. You can inspect the individual coefficients and other 
relevant objects by inspecting the fields of the struct. For example, if 
you have 

    sol = stepwise_selection(cell_sol; kwargs...)

then `sol.diffusion_theta` will be the set of coefficients for the diffusion 
function. You can query the full list of fields by using 
    
    propertynames(sol)

or typing `sol.<TAB>` in the REPL (`<TAB>` means hit tab). We also note that if you 
want to have more interactivity with the table that gets displayed in the REPL, you should 
look into the `show` method for `EQLSolution`, which has signature 

    Base.show(io::IO, ::MIME"text/plain", eql_sol::EQLSolution;
        step_limit=6,
        crop=:horizontal,
        backend=Val(:text),
        booktabs=true,
        show_votes=false,
        show_all_loss=false,
        crayon=Crayon(bold=true, foreground=:green),
        latex_crayon=["color{blue}", "textbf"],
        transpose=true)

For example,

    Base.show(stdout, MIME"text/plain"(), sol; show_votes=false, show_all_loss=false, transpose=true)

prints the table without including the `votes`, only shows the complete loss function rather than also 
including its individual components, and transposes the table. A LaTeX version of the table can be obtained 
using `backend=Val(:latex)`, e.g. a LaTeX version of the above could be printed using

    Base.show(stdout, MIME"text/plain"(), sol; backend=Val(:latex), show_votes=false, show_all_loss=false, transpose=true)

You can also just use 

    latex_table(sol; kwargs...)

to get the LaTeX format printed.

- `EnsembleEQLSolution`

In this case, you have provided `model_samples` as a keyword argument and 
`model_samples > 1`. This struct has four fields: `solutions`, which stores 
all the individual `EQLSolutions`; `final_loss`, which stores a `Dict` mapping 
final vectors of active coefficients to `Tuples` of the form `(loss, n)`, where 
`loss` is the loss function at that final model, and `n` is the number of times that 
model was found out of the complete set of initial indicator vectors sampled; 
`best_model`, which is an indicator vector which gives the model out of those found with the
least loss; `best_model_indices`, which gives the indices of all solutions in `solutions` which had a 
final indicator vector matching that of `best_model`.

# Keyword Arguments 
- `(diffusion/reaction/rhs/moving_boundary)_basis::BasisSet`

The basis expansion to use for the diffusion, reaction, right-hand side, and
moving boundary terms, respectively. If not provided, the functions are replaced 
with the zero function. For `diffusion_basis`, a `BasisSet` is required.
- `(diffusion/reaction/rhs/moving_boundary)_parameters`

The parameters to use for evaluating the corresponding basis set. The default 
is `nothing`.
- `(diffusion/reaction/rhs/moving_boundary)_theta`

This keyword argument indicates whether the coefficients `θ` used for evaluating the basis sets 
are fixed or are to be learned, with a defualt of `nothing`. If `nothing`, then they will be learned. Otherwise, they should be 
a vector of numbers indicating the coefficients `θ` to be used for the respective mechanism. For example,
`diffusion_theta = [1.0, 0.5]` means that the diffusion function is evaluated with coefficients 
`θ₁ᵈ = 1` and `θ₂ᵈ = 0.5`.
- `threshold_tol`

The threshold tolerances to use for pruning the matrix, with a default of a zero tolerance for each 
quantity. This should be a `NamedTuple` if provided, only including any of the names `q`, `x`, 
`dt`, `dx`, `dx2`, `dx_bc`, and `dL`. The `q`, `dt`, `dx`, and `dx2` tolerances are used for pruning 
the interior PDE, corresponding to tolerances based on the quantiles of `q(x, t)`, `∂q(x, t)/∂t`,
`∂q(x, t)/∂x`, and `∂²q(x, t)/∂x²`. The `dx_bc` tolerance is used for pruning based on `∂q(x, t)/∂t` evaluated 
at the boundary, and is only used for the matrix corresponding to the right-hand side boundary condition. The 
`dL` tolerance is for `dL/dt`, and is only used for the matrix corresponding to the moving boundary condition. 
For example, providing 

    threshold_tol = (q = 0.1, dL = 0.3, x = 0.7, dx = 0.1, dx_bc = 0.05)

means that the matrix for the interior problem will only include points whose densities are between the 
10% and 90% density quantiles, the 10% and 90% quantiles for `∂q(x, t)/∂x`, at positions between `0`
and 70% of the current leading edge for the given time. The matrix for the right-hand side boundary condition 
will only include points between the 5% and 95% quantiles for `∂q(x, t)/∂x` at the boundary. The matrix 
for the moving boundary condition will only include points where the velocity `dL/dt` is between the 30% 
and 70% quantiles of `dL/dt`.
- `mesh_points`

The number of mesh points to use for discretising the corresponding PDE. The default is `100`.
- `cross_validation`

Whether to use cross-validation for selecting the model. The default is `false`.
- `rng`

The random number generator to use. This defaults to the global RNG, `Random.default_rng()`.
- `skip`

Indices of any coefficients that should be skipped over when deciding which terms 
to include or remove. By default, this is empty
- `regression`

Whether to include the regression loss in the loss function. Defaults to `false`.
- `density`

Whether to include the density loss in the loss function. Defaults to `true`.
- `complexity`

The complexity penalty to use for the loss function. Defaults to `1`. Use `0` if you would 
not like to penalise complexity.
- `loss_function`

The loss function use. Defaults to `default_loss(; regression, density, complexity)`. The loss 
function should take the same form as described in `?default_loss`.
- `bidirectional`

Whether to allow for steps in both directions, meaning terms can be either added or deleted at each step. 
The default is `true`. If `false`, then steps can only be taken backwards.
- `trials`

If `cross_validation` is true, this is the number of votes to attempt at each step of the algorithm. 
- `initial`

The initial set of active coefficients. The default is `:all`, meaning all terms are initially 
active. If `:random`, then a random set of terms is initially active. If `:none`,
then all terms are initially inactive. Otherwise, `initial` should be a vector of Booleans indicating the 
initially active terms, with `true` for active and `false` for inactive. The indices should match 
the flattened form of the coefficients, with the order of coefficients being `diffusion`, `reaction`, 
`rhs`, and then `moving_boundary`. For example, if the basis functions are

    D(q) = θ₁ᵈϕ₁ᵈ + θ₂ᵈϕ₂ᵈ                      (diffusion)
    R(q) = θ₁ʳϕ₁ʳ + θ₂ʳϕ₂ʳ + θ₃ʳϕ₃ʳ             (reaction)
    H(q) = θ₁ʰϕ₁ʰ + θ₂ʰϕ₂ʰ + θ₃ʰϕ₃ʰ + θ₄ʰϕ₄ʰ    (rhs)
    E(q) = θ₁ᵉϕ₁ᵉ + θ₂ᵉϕ₂ᵉ                      (moving boundary)

and you want to start with `θ₁ᵈ`, `θ₁ʳ`, `θ₂ʳ`, `θ₂ʰ`, and `θ₂ᵉ` active, 
then you should provide 

    initial = [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1]
- `model_samples`

How many initial models to randomly sample when providing the results. This defaults to `1`.
If `model_samples > 1`, then `initial` must be `:random`. The result returned in this case 
will be an `EnsembleEQLSolution`.
- `use_relative_err`

Whether to use relative error (`true`) or absolute error (`false)` in the loss function summands. Defaults 
to `true`.
- `aggregate`

When considering an ensemble of cell solutions, `aggregate=true` indicates that the matrix constructed 
should combine all the cell solutions into one matrix, and the loss function should be the sum of the
loss functions for each cell solution. The default is `false`. If this is `false`, then for an ensemble of cell 
solutions you must have `average=Val(true)`; if both are `false` and the cell solution is an ensemble, the call will error. 
- `time_range`

The time range, provided as a `Tuple` of the form `(tmin, tmax)`, to use for constructing the matrix. 
The default is the complete time span that the cell simulation is performed over. 
- `average`

When considering an ensemble of cell solutions, `average=Val(true)` indicates that the matrix constructed 
should average the solutions together into a single matrix, and the loss function should use this averaged solution, 
given by an `AveragedODESolution`. The default is `Val(true)`. If this is `Val(false)`, then for an ensemble of cell 
solutions you must have `aggregate=true`; if both are `false` and the cell solution is an ensemble, the call will error. 
- `simulation_indices`

The simulations to consider in the matrix. Only relevant if an ensemble of cell solutions is being considered.
Defaults to all simulations.
- `num_knots`

The number of knots to use in the interpolant for averaging the cell solutions. Only relevant if an ensemble of cell solutions is being considered.
Defaults to `100`.
- `avg_interp_fnc`

A function of the form `(u, t)` that is used for averaging. Only relevant if an ensemble of cell solutions is being considered.
Defaults to linear interpolation from DataInterpolations.jl.
- `stat`

This should be a function, or a `Tuple` of two functions with one for each endpoint, that decides the range to use 
for the knots at each time step. Only relevant if an ensemble of cell solutions is being considered. Defaults to `mean`, 
meaning the knots for a given time will span between `a` and `b`, where `a` is the average left-most cell position over each 
simulation at that time, and `b` is the average right-most cell position over each simulation at that time. If, for example,
`stat` were `(minimum, maximum)`, then the minimum and maximum end-positions would be used for `a` and `b`, respectively, 
at the corresponding time.
- `show_progress`

Whether to print the current step of the algorithm at each step. Defaults to `true`.
- `max_steps`

Maximum number of steps to allow for the algorithm. Defaults to `100`.
- `leading_edge_error`

Whether to include the leading edge error in the loss function. If this is `true`,
then `density` must also be `true`. Only relevant if the cell simulation being considered is a moving boundary 
problem. Defaults to `true`.
- `extrapolate_pde`

If the time range is smaller than the time span of the cell simulation, then setting `extrapolate_pde`
to `true` means that the loss function will still consider all saved time points, not just those within the provided 
`time_range`. The default is `false`.
- `num_constraint_checks`

The number of equally-spaced nodes to use between the minimum and maximum densities for constraining the diffusion term and the moving boundary term.
Defaults to `100`.
- `conserve_mass`

Whether to enforce conservation of mass by constraining the diffusion and moving boundary terms to be equal. Defaults to `false`.
- `extrapolate_average`

Whether to allow for extrapolation when averaging the cell solutions. Only relevant if an ensemble of cell solutions is being considered.
Defaults to `true`. If `false`, then when evaluating the interpolant for a given realisation and a given time, then the density for a node 
is set to zero if it exceeds the right-most knot position for the given time.
- `couple_rhs`

Whether to couple the boundary conditions at the right-hand side and at the moving boundary together. Only relevant if the moving boundary coefficients 
are fixed. If `true`, then the matrix for the right-hand boundary condition will also include terms coming from the moving boundary, 
replacing `∂q/∂x` with the right-hand side's assumed basis expansion. Defaults to `true`. If `false`,
then the matrix for the right-hand boundary condition will only include terms coming from the right-hand side.
"""
function stepwise_selection(cell_sol;
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
    mesh_points=100,
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
    cross_validation=false,
    rng=Random.default_rng(),
    skip=(),
    regression=false,
    density=true,
    complexity=1,
    loss_function=default_loss(; regression, density, complexity),
    bidirectional=true,
    trials=25,
    initial=:all, # must be = :random if model_samples > 1 
    model_samples=1,
    use_relative_err=true,
    aggregate=false,
    time_range=default_time_range(cell_sol),
    average::Val{B}=Val(true),
    simulation_indices=eachindex(cell_sol),
    num_knots=100,
    avg_interp_fnc=LinearInterpolation{true},
    stat=mean,
    show_progress=false,
    max_steps=100,
    leading_edge_error=true,
    extrapolate_pde=false,
    num_constraint_checks=100,
    conserve_mass=false,
    extrapolate_average=true,
    couple_rhs=false) where {B}
    if conserve_mass
        moving_boundary_basis = diffusion_basis
        moving_boundary_parameters = diffusion_parameters
        moving_boundary_theta = diffusion_theta
    end
    time_range = check_args(; cell_sol, time_range, aggregate, B, density, leading_edge_error, diffusion_theta, conserve_mass)
    model = EQLModel(cell_sol;
        diffusion_basis, reaction_basis, rhs_basis, moving_boundary_basis,
        diffusion_parameters, reaction_parameters, rhs_parameters, moving_boundary_parameters,
        diffusion_theta, reaction_theta, rhs_theta, moving_boundary_theta,
        threshold_tol, pde, time_range,
        average, simulation_indices, num_knots, avg_interp_fnc,
        stat, conserve_mass, extrapolate_average, couple_rhs)
    cell_sol = model.cell_sol
    if model_samples == 1 && (cell_sol isa ODESolution || cell_sol isa AveragedODESolution || aggregate)
        return _stepwise_selection_single(model, initial, cross_validation, rng, skip, loss_function, bidirectional, trials, use_relative_err, show_progress, max_steps, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)
    elseif model_samples > 1 && (cell_sol isa ODESolution || cell_sol isa AveragedODESolution || aggregate)
        return _stepwise_selection_ensemble(model, initial, cross_validation, rng, skip, loss_function, bidirectional, trials, model_samples, use_relative_err, aggregate, max_steps, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)
    else
        throw("Error occurred. Please double check your keyword arguments or report an issue.")
    end
end