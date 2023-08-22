function stepwise_selection(model::EQLModel, indicators=model.indicators;
    cross_validation=false,
    rng=Random.default_rng(),
    skip=(),
    regression=false,
    density=true,
    complexity=1,
    loss_function=default_loss(; regression, density, complexity),
    bidirectional=true,
    trials=100,
    use_relative_err=true,
    show_progress=false,
    max_steps=100,
    leading_edge_error=true,
    extrapolate_pde=false,
    num_constraint_checks=100,
    conserve_mass=false)
    model_changed = true
    best_model = 0
    indicator_history = ElasticMatrix{Bool}(undef, length(indicators), 0)
    vote_history = ElasticMatrix{Float64}(undef, length(indicators) + 1, 0)
    append!(indicator_history, copy(indicators))
    ctr = 0
    while model_changed
        ctr += 1
        show_progress && println("Starting step $ctr.")
        model_changed, best_model, votes = step!(model, indicators; cross_validation, rng, loss_function, bidirectional, trials, skip=union(best_model, skip), use_relative_err, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)
        model_changed && ctr < max_steps && append!(indicator_history, copy(indicators))
        append!(vote_history, votes)
        if ctr ≥ max_steps
            @warn "Maximum number of steps reached. Returning current solution."
            break
        end
    end
    return EQLSolution(model, indicator_history, vote_history, loss_function; use_relative_err, leading_edge_error, extrapolate_pde, conserve_mass)
end

function check_args(; cell_sol, time_range, aggregate, B, density, leading_edge_error, diffusion_theta, conserve_mass)
    t = cell_sol isa EnsembleSolution ? cell_sol[1].t : cell_sol.t
    if time_range[1] == t[1]
        time_range = (t[2], oftype(t[2], time_range[2]))
        @warn "Lower time bound of time_range is equal to the first time in cell_sol which is not allowed. Setting lower time bound to $(t[2]), the second time."
    end
    @assert issorted(time_range) "time_range must be sorted"
    @assert allunique(get_saveat(cell_sol)) "cell_sol must have unique times"
    @assert issorted(get_saveat(cell_sol)) "cell_sol times must be sorted"
    if cell_sol isa EnsembleSolution
        @assert xor(aggregate, B) == 1 "Must choose at least one of aggregate or average (but not both)."
    end
    if leading_edge_error
        @assert density "The leading edge loss requires density to be true so that the density loss is included."
    end
    if !isnothing(diffusion_theta) && conserve_mass
        throw(ArgumentError("conserve_mass cannot be true if D(q) is fixed."))
    end
    return time_range
end

function get_indicators(model, initial, rng)
    nind = length(model.indicators)
    indicators = zeros(Bool, nind)
    if initial == :all
        indicators .= true
    elseif initial == :none
        indicators .= false
    elseif initial == :random
        indicators .= rand(rng, Bool, nind)
    elseif initial isa AbstractVector
        @assert length(initial) == length(model.indicators) "initial must be a Boolean vector of length $(length(model.indicators))"
        indicators .= initial
    else
        throw(ArgumentError("initial must be one of :all, :none, :random, or an AbstractVector"))
    end
    return indicators
end

function _stepwise_selection_single(model::EQLModel, initial, cross_validation, rng, skip, loss_function, bidirectional, trials, use_relative_err, show_progress, max_steps, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)
    indicators = get_indicators(model, initial, rng)
    eql_sol = stepwise_selection(model, indicators; cross_validation, rng, skip, loss_function, bidirectional, trials, use_relative_err, show_progress, max_steps, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)
    return eql_sol
end

