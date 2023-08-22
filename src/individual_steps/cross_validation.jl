num_time_points(sol::Union{ODESolution,AveragedODESolution}) = length(sol.t)
num_time_points(sol::EnsembleSolution) = num_time_points(sol[1])
num_simulations(sol::Union{ODESolution,AveragedODESolution}) = 1
num_simulations(sol::EnsembleSolution) = length(sol)
function get_training_and_test_subsets(model; extrapolate_pde=false, cross_validation=false, rng=Random.default_rng())
    nt = length(model.valid_time_indices)
    if !cross_validation
        if !extrapolate_pde
            return (model.valid_time_indices, (collect ∘ axes)(model.A, 1)), (model.valid_time_indices, (collect ∘ axes)(model.A, 1)), model.valid_time_indices
        else
            return (model.valid_time_indices, (collect ∘ axes)(model.A, 1)), (model.valid_time_indices, (collect ∘ axes)(model.A, 1)), 2:num_time_points(model.cell_sol)
        end
    end
    training_times = sample(rng, model.valid_time_indices, ceil(Int, 0.8nt), replace=false, ordered=true) # Having sorted indices helps with contiguous array access
    training_subset = Int[]
    sizehint!(training_subset, size(model.A, 1) ÷ 2)
    ns = num_simulations(model.cell_sol)
    sort_flag = false
    for k in 1:ns
        cell_sol = model.cell_sol isa EnsembleSolution ? model.cell_sol[k] : model.cell_sol
        for j in training_times
            r = cell_sol.u[j]
            for i in eachindex(r)
                if (k, i, j) ∈ model.idx_map.range
                    push!(training_subset, model.idx_map((k, i, j)))
                end
            end
            if (k, 0, j) ∈ model.idx_map.range
                sort_flag = true
                push!(training_subset, model.idx_map((k, 0, j)))
            end
            if (k, -1, j) ∈ model.idx_map.range
                sort_flag = true
                push!(training_subset, model.idx_map((k, -1, j)))
            end
        end
    end
    sort_flag && sort!(training_subset)
    test_subset = setdiff(axes(model.A, 1), training_subset)
    test_times = setdiff(model.valid_time_indices, training_times)
    if !extrapolate_pde
        pde_times = test_times
    else
        _nt = num_time_points(model.cell_sol)
        _training_times = sample(rng, 2:_nt, ceil(Int, 0.8 * _nt), replace=false, ordered=true)
        pde_times = setdiff(2:_nt, _training_times)
    end
    return (training_times, training_subset), (test_times, test_subset), pde_times
end