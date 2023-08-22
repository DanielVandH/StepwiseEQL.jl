
"""
    AveragedODESolution{K,S}

Struct representing a cell simulation averaged over many simulations. 

# Fields 
- `u::K`

These are the knots, storing the knots using for interpolating the solutions 
at each time, where `u[j]` the knots for the `j`th time. 
- `t::Vector{Float64}`

The vector of times.
- `q::Vector{Vector{Float64}}`

The averaged densities, with `q[j]` the averaged density at time `t[j]` and 
at the corresponding knots in `u[j]`.
- `cell_sol::S`

The original cell simulation.

# Constructor
You can construct an `AveragedODESolution` using the following method:

    AveragedODESolution(
        sol::EnsembleSolution, 
        num_knots=100, 
        indices=eachindex(sol), 
        interp_fnc=LinearInterpolation, 
        stat=mean, 
        extrapolate=true
    )

Note that the arguments are positional arguments, not keyword arguments.

## Arguments 
- `sol`

The ensemble of cell solutions.
- `num_knots`

The number of knots to use in the interpolant for averaging the cell solutions. 
Defaults to `100`.
- `indices`

The indices of the simulations to consider for averaging. Defaults to all simulations.
- `interp_fnc`

A function of the form `(u, t)` that is used for averaging. 
Defaults to linear interpolation from DataInterpolations.jl.
- `stat`

This should be a function, or a `Tuple` of two functions with one for each endpoint, that decides the range to use 
for the knots at each time step. Defaults to `mean`, 
meaning the knots for a given time will span between `a` and `b`, where `a` is the average left-most cell position over each 
simulation at that time, and `b` is the average right-most cell position over each simulation at that time. If, for example,
`stat` were `(minimum, maximum)`, then the minimum and maximum end-positions would be used for `a` and `b`, respectively, 
at the corresponding time.
- `extrapolate=true`

Whether to allow for extrapolation when averaging the cell solutions.
Defaults to `true`. If `false`, then when evaluating the interpolant for a given realisation and a given time, then the density for a node 
is set to zero if it exceeds the right-most knot position for the given time.

## Output 
Returns the corresponding `AveragedODESolution`.
"""
struct AveragedODESolution{K,S}
    u::K                        # knots 
    t::Vector{Float64}          # times
    q::Vector{Vector{Float64}}  # densities 
    cell_sol::S
end

@memoize function AveragedODESolution(sol::EnsembleSolution, num_knots=100, indices=eachindex(sol), interp_fnc=LinearInterpolation{true}, stat=mean, extrapolate=true)
    (; means, knots) = node_densities_means_only(sol; num_knots, indices, interp_fnc, stat, extrapolate)
    return AveragedODESolution(knots, sol[begin].t, means, sol)
end
Base.length(sol::AveragedODESolution) = length(sol.q)
Base.lastindex(sol::AveragedODESolution) = lastindex(sol.q)
Base.eachindex(sol::AveragedODESolution) = eachindex(sol.q)
