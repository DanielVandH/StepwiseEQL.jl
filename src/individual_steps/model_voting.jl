function _add1!(results, term, model, indicators, loss_function, subsets, skip, use_relative_err, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)
    flag = indicators[term]
    if !flag && term ∉ skip
        indicators[term] = true
        results[term] = evaluate_model_loss(model, indicators, loss_function; subsets, use_relative_err, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)::Float64
    end
    return nothing
end
function _drop1!(results, term, model, indicators, loss_function, subsets, skip, use_relative_err, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)
    flag = indicators[term]
    if flag && term ∉ skip
        indicators[term] = false
        results[term] = evaluate_model_loss(model, indicators, loss_function; subsets, use_relative_err, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)::Float64
    end
    return nothing
end

@inline function add1(model::EQLModel, indicators=model.indicators, loss_function=default_loss(; regression=false);
    cross_validation=false,
    rng=Random.default_rng(),
    extrapolate_pde=false,
    subsets=get_training_and_test_subsets(model; extrapolate_pde, cross_validation, rng),
    skip=(),
    use_relative_err=true,
    leading_edge_error=true,
    num_constraint_checks=100,
    conserve_mass=false)
    results = Dict{Int,Float64}()
    @sync for term in eachindex(indicators)
        Threads.@spawn _add1!(results, term, model, deepcopy(indicators), loss_function, subsets, skip, use_relative_err, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)
    end
    return results
end

@inline function drop1(model::EQLModel, indicators=model.indicators, loss_function=default_loss(; regression=false);
    cross_validation=false,
    rng=Random.default_rng(),
    extrapolate_pde=false,
    subsets=get_training_and_test_subsets(model; extrapolate_pde, cross_validation, rng),
    skip=(),
    use_relative_err=true,
    leading_edge_error=true,
    num_constraint_checks=100,
    conserve_mass=false)
    results = Dict{Int,Float64}()
    @sync for term in eachindex(indicators)
        Threads.@spawn _drop1!(results, term, model, deepcopy(indicators), loss_function, subsets, skip, use_relative_err, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)
    end
    return results
end

function vote(model::EQLModel, subsets, indicators, loss_function=default_loss(; regression=false); skip=(), bidirectional=true, use_relative_err=true, leading_edge_error=true, extrapolate_pde=false, num_constraint_checks=100, conserve_mass=false)
    possible_models = Dict{Int,Float64}()
    if bidirectional
        begin
            backward_possible_models = Threads.@spawn drop1(model, indicators, loss_function; subsets, skip, use_relative_err, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)
            forward_possible_models = Threads.@spawn add1(model, indicators, loss_function; subsets, skip, use_relative_err, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)
            backward_possible_models = fetch(backward_possible_models)
            forward_possible_models = fetch(forward_possible_models)
            merge!(possible_models, backward_possible_models, forward_possible_models)
        end
    else
        backward_possible_models = drop1(model, indicators, loss_function; subsets, skip, use_relative_err, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)
        merge!(possible_models, backward_possible_models)
    end
    if length(possible_models) > 0
        minimum_score, best_model = findmin(possible_models)
    else
        minimum_score, best_model = Inf, 0
    end
    if count(indicators) > 0
        current_score = evaluate_model_loss(model, indicators, loss_function; subsets, use_relative_err, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)
    else
        current_score = Inf
    end
    model_changed = minimum_score < current_score
    votes = zeros(Int64, length(indicators) + 1)
    if model_changed
        votes[best_model] = indicators[best_model] ? -1 : 1 # -1 to signify that the term was dropped
    else
        votes[end] += 1
    end
    return votes
end

function vote(model::EQLModel, indicators;
    cross_validation=false,
    rng=Random.default_rng(),
    skip=(),
    loss_function=default_loss(; regression=false),
    bidirectional=true,
    trials=100,
    use_relative_err=true,
    leading_edge_error=true,
    extrapolate_pde=false,
    num_constraint_checks=100,
    conserve_mass=false)
    trials = !cross_validation ? 1 : trials
    @floop FLOOPS_EX for _ in 1:trials
        subsets = get_training_and_test_subsets(model; cross_validation, extrapolate_pde, rng)
        _votes = vote(model, subsets, indicators, loss_function; skip, bidirectional, use_relative_err, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)
        @reduce(votes = zeros(length(indicators) + 1) + _votes)
    end
    votes ./= trials
    _, best_model = findmax(abs, votes)
    return best_model, votes
end

function step!(model::EQLModel, indicators=model.indicators;
    cross_validation=false,
    rng=Random.default_rng(),
    skip=(),
    loss_function=default_loss(; regression=false),
    bidirectional=true,
    trials=100,
    use_relative_err=true,
    leading_edge_error=true,
    extrapolate_pde=false,
    num_constraint_checks=100,
    conserve_mass=false)
    best_model, votes = vote(model, indicators; cross_validation, rng, skip, loss_function, bidirectional, trials, use_relative_err, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)
    model_changed = best_model ≠ length(indicators) + 1 # length(indicators) + 1 is the "no change" model
    if model_changed
        indicators[best_model] = !indicators[best_model]
    end
    return model_changed, best_model, votes
end