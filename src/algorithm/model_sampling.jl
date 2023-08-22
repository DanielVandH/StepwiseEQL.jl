function _stepwise_selection_ensemble(model::EQLModel, initial, cross_validation, rng, skip, loss_function, bidirectional, trials, model_samples, use_relative_err, aggregate, max_steps, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)
    if (model.cell_sol isa ODESolution || aggregate) && initial ≠ :random
        throw(ArgumentError("initial must be :random when model_samples > 1."))
    end
    if (model.cell_sol isa ODESolution || aggregate) && model_samples ≥ 2^length(model.indicators) && !cross_validation # If !cross_validation, then a deterministic loss is produced for each initial indicator rather than a stochastic one
        model_samples = 2^length(model.indicators)
        indicators = vec([collect(v) for v in Iterators.product(fill([false, true], length(model.indicators))...)]) # all possible combinations of true and false. Yes, there is a better way to do this. No, it's not important since this is not where performance matters.
        @warn "model_samples ≥ 2^length(model.indicators). Setting model_samples = 2^length(model.indicators) = $(model_samples) and using all possible combinations of true and false as initial conditions."
    else
        indicators = [get_indicators(model, initial, rng) for _ in 1:model_samples]
    end
    if (model.cell_sol isa ODESolution || aggregate) && !cross_validation
        unique!(indicators)
        model_samples = length(indicators)
    end
    solutions = Vector{Any}(undef, model_samples) # Could use Folds.mapreduce here to make the eltype obtained in a smart way, without having to compute one solution outside of the loop, but it doesn't really matter for performance (since so many other things take a long time)
    Base.Threads.@threads for i in 1:model_samples
        solutions[i] = stepwise_selection(model, indicators[i]; cross_validation, rng, skip, loss_function, bidirectional, trials, use_relative_err, max_steps, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)
    end
    return EnsembleEQLSolution(solutions)
end