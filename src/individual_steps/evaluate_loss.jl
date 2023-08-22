"""
    default_loss(; kwargs...)
    default_loss(regression_loss, density_loss, indicators; density=true, regression=false, complexity=1)

The default loss function. The first method, accepting the same keyword arguments as in the second method, returns 
a function that accepts the regression loss, density loss, and indicators, and returns the total loss, evaluated 
via the second method.

# Arguments 
- `regression_loss`: The value of the regression loss.
- `density_loss`: The value of the density loss.
- `indicators`: A vector of Boolean values, indicating which the set of active coefficients.

# Keyword Arguments
- `density=true`: Whether to include the density loss.
- `regression=false`: Whether to include the regression loss.
- `complexity=1`: The complexity penalty coefficient. Use `0` to disable the complexity penalty.
"""
default_loss(; kwargs...) =
    let kwargs = kwargs
        @inline (regression_loss, density_loss, indicators) -> default_loss(regression_loss, density_loss, indicators; kwargs...)
    end
@inline function default_loss(regression_loss, density_loss, indicators; density=true, regression=false, complexity=1)
    loss = 0.0
    regression && (loss += regression_loss)
    density && (loss += density_loss)
    loss += parsimony_penalty(indicators, complexity)
    return loss
end

function evaluate_loss(model, indicators=model.indicators;
    cross_validation=false,
    rng=Random.default_rng(),
    extrapolate_pde=false,
    subsets=get_training_and_test_subsets(model; extrapolate_pde, cross_validation, rng),
    use_relative_err=true,
    leading_edge_error=true,
    num_constraint_checks=100,
    conserve_mass=false)
    (_, training_subset), (test_times, test_subset), pde_times = subsets
    θ = projected_solve(model, training_subset, indicators)
    diffusion_or_moving_boundary_is_negative = check_negative_diffusion_or_moving_boundary(model, θ, num_constraint_checks, conserve_mass)
    if diffusion_or_moving_boundary_is_negative
        return Inf, Inf
    end
    regression_loss = evaluate_regression_loss(model, θ, test_subset, indicators; use_relative_err)::Float64
    density_loss = evaluate_density_loss(model, θ, pde_times, indicators; use_relative_err, leading_edge_error, conserve_mass)::Float64
    return isfinite(regression_loss) ? regression_loss : Inf, isfinite(density_loss) ? density_loss : Inf
end

function evaluate_model_loss(model, indicators=model.indicators, loss_function=default_loss(; regression=false);
    cross_validation=false,
    rng=Random.default_rng(),
    extrapolate_pde=false,
    subsets=get_training_and_test_subsets(model; extrapolate_pde, cross_validation, rng),
    use_relative_err=true,
    leading_edge_error=true,
    num_constraint_checks=100,
    conserve_mass=false)
    regression_loss, density_loss = evaluate_loss(model, indicators; cross_validation, rng, subsets, use_relative_err, leading_edge_error, extrapolate_pde, num_constraint_checks, conserve_mass)
    tot_loss = loss_function(regression_loss, density_loss, indicators)
    if !isfinite(tot_loss) # also checks for NaN
        return Inf
    else
        return tot_loss
    end
end