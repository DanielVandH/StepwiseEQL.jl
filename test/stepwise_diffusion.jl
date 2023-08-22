using ..StepwiseEQL
using EpithelialDynamics1D
using OrdinaryDiffEq
using LinearSolve
using StatsBase
using CairoMakie
using ReferenceTests
using LinearAlgebra
using Random
using Printf
using Setfield
using FiniteVolumeMethod1D
using NaNMath
using ElasticArrays
const EQL = StepwiseEQL
fig_path = normpath(@__DIR__, "..", "test", "figures")

#### Diffusion 
### Setup 
LogRange(a, b, n) = exp10.(LinRange(log10(a), log10(b), n))

force_law = (δ, p) -> p.k * (p.s - δ)
force_law_parameters = (k=50.0, s=0.2)
final_time = 5.0
damping_constant = 1.0
initial_condition = [LinRange(0, 15, 16); LinRange(15, 30, 32)] |> unique!
prob = CellProblem(;
    force_law,
    force_law_parameters,
    final_time,
    damping_constant,
    initial_condition)
sol = solve(prob, Tsit5(), saveat=LogRange(1e-16, 5.0, 1000))
diffusion_basis = BasisSet(
    (u, k) -> inv(u),
    (u, k) -> inv(u^2),
    (u, k) -> inv(u^3),
    (u, k) -> inv(u^4)
)

### Learning parameters: No cross-validation and no indicator sampling 
cell_sol = sol
diffusion_basis = diffusion_basis
reaction_basis = BasisSet()
diffusion_parameters = nothing
reaction_parameters = nothing
diffusion_theta = nothing
reaction_theta = nothing
cross_validation = false
bidirectional = true
mesh_points = 250
threshold_tol = (q=1e-3, dt=1e-3)
density = true
regression = false
loss_function = default_loss(; density, regression)
trials = 100
skip = ()
initial = copy([true, true, true, true])
rng = Random.seed!(1234)
eql_sol = stepwise_selection(sol;
    diffusion_basis,
    reaction_basis,
    diffusion_parameters,
    reaction_parameters,
    diffusion_theta,
    reaction_theta,
    cross_validation,
    bidirectional,
    mesh_points,
    threshold_tol,
    density,
    regression,
    loss_function,
    trials,
    skip,
    initial,
    rng
)

## Setting up the initial model 
pde = EQL.build_pde(cell_sol,
    mesh_points;
    diffusion_basis,
    reaction_basis,
    diffusion_parameters,
    reaction_parameters,
    diffusion_theta,
    reaction_theta)
model = EQL.EQLModel(cell_sol;
    diffusion_basis,
    reaction_basis,
    diffusion_parameters,
    reaction_parameters,
    diffusion_theta,
    reaction_theta,
    threshold_tol,
    mesh_points,
    pde)
model.indicators .= initial
model_changed = true
best_model = 0
indicator_history = ElasticMatrix{Bool}(undef, length(model.indicators), 0)
vote_history = ElasticMatrix{Float64}(undef, length(model.indicators) + 1, 0)
append!(indicator_history, copy(model.indicators))

## Start stepping
# Step 1
rng = Random.seed!(1234)
model_changed, best_model, votes = EQL.step!(model; cross_validation, rng, loss_function, bidirectional, trials, skip=union(best_model, skip))
model.indicators[best_model] = !model.indicators[best_model] # undo so we can test 
all_loss = zeros(length(model.indicators) + 1)
for i in eachindex(all_loss)
    if i ≠ lastindex(all_loss)
        model.indicators[i] = !model.indicators[i]
    end
    all_loss[i] = EQL.evaluate_model_loss(
        model,
        model.indicators,
        default_loss(; density, regression),
        cross_validation=cross_validation,
        rng=rng
    )
    if i ≠ lastindex(all_loss)
        model.indicators[i] = !model.indicators[i] # undo the change to finish the loop
    end
end
min_resid = findlast(all_loss .== minimum(all_loss))
@test min_resid == best_model
model.indicators[best_model] = !model.indicators[best_model] # redo 
append!(indicator_history, copy(model.indicators))
append!(vote_history, votes)

# Step 2 
rng = Random.seed!(1234)
model_changed, best_model, votes = EQL.step!(model; cross_validation, rng, loss_function, bidirectional, trials, skip=union(best_model, skip))
model.indicators[best_model] = !model.indicators[best_model] # undo so we can test 
all_loss = zeros(length(model.indicators) + 1)
for i in eachindex(all_loss)
    if i ≠ lastindex(all_loss)
        model.indicators[i] = !model.indicators[i]
    end
    all_loss[i] = EQL.evaluate_model_loss(
        model,
        model.indicators,
        default_loss(; density, regression),
        cross_validation=cross_validation,
        rng=rng
    )
    if i ≠ lastindex(all_loss)
        model.indicators[i] = !model.indicators[i] # undo the change to finish the loop
    end
end
min_resid = findlast(all_loss .== minimum(all_loss))
@test min_resid == best_model
model.indicators[best_model] = !model.indicators[best_model] # redo 
append!(indicator_history, copy(model.indicators))
append!(vote_history, votes)

# Step 3 
rng = Random.seed!(1234)
model_changed, best_model, votes = EQL.step!(model; cross_validation, rng, loss_function, bidirectional, trials, skip=union(best_model, skip))
model.indicators[best_model] = !model.indicators[best_model] # undo so we can test 
all_loss = zeros(length(model.indicators) + 1)
for i in eachindex(all_loss)
    if i ≠ lastindex(all_loss)
        model.indicators[i] = !model.indicators[i]
    end
    all_loss[i] = EQL.evaluate_model_loss(
        model,
        model.indicators,
        default_loss(; density, regression),
        cross_validation=cross_validation,
        rng=rng
    )
    if i ≠ lastindex(all_loss)
        model.indicators[i] = !model.indicators[i] # undo the change to finish the loop
    end
end
min_resid = findlast(all_loss .== minimum(all_loss))
@test min_resid == best_model
model.indicators[best_model] = !model.indicators[best_model] # redo 
append!(indicator_history, copy(model.indicators))
append!(vote_history, votes)

# Step 4 
rng = Random.seed!(1234)
model_changed, best_model, votes = EQL.step!(model; cross_validation, rng, loss_function, bidirectional, trials, skip=union(best_model, skip))
@test !model_changed
@test best_model == 5
all_loss = zeros(length(model.indicators) + 1)
for i in eachindex(all_loss)
    if i ≠ lastindex(all_loss)
        model.indicators[i] = !model.indicators[i]
    end
    all_loss[i] = EQL.evaluate_model_loss(
        model,
        model.indicators,
        default_loss(; density, regression),
        cross_validation=cross_validation,
        rng=rng
    )
    if i ≠ lastindex(all_loss)
        model.indicators[i] = !model.indicators[i] # undo the change to finish the loop
    end
end
min_resid = findlast(all_loss .== minimum(all_loss))
@test min_resid == best_model
append!(vote_history, votes)

# Test the structs are matching 
_eql_sol = EQL.EQLSolution(model, indicator_history, vote_history, loss_function)
@test _eql_sol.diffusion_theta == eql_sol.diffusion_theta
@test _eql_sol.reaction_theta == eql_sol.reaction_theta
@test _eql_sol.diffusion_subset == eql_sol.diffusion_subset
@test _eql_sol.reaction_subset == eql_sol.reaction_subset
@test _eql_sol.diffusion_theta_history == eql_sol.diffusion_theta_history
@test _eql_sol.reaction_theta_history == eql_sol.reaction_theta_history
@test _eql_sol.diffusion_vote_history == eql_sol.diffusion_vote_history
@test _eql_sol.reaction_vote_history == eql_sol.reaction_vote_history
@test _eql_sol.density_loss_history == eql_sol.density_loss_history
@test _eql_sol.loss_history == eql_sol.loss_history
@test _eql_sol.indicator_history == eql_sol.indicator_history
@test _eql_sol.vote_history == eql_sol.vote_history
@test _eql_sol.pde_sol.u ≈ eql_sol.pde_sol.u

# Test the actual values of the struct
indicators = [false, true, false, false]
θ = model.A[:, indicators] \ model.b
diffusion_theta = [0.0, θ[1], 0.0, 0.0]
diffusion_subset = [2]
reaction_theta = Float64[]
reaction_subset = Int64[]
θ1 = model.A \ model.b
θ2 = model.A[:, [false, true, true, true]] \ model.b
θ3 = model.A[:, [false, true, true, false]] \ model.b
θ4 = model.A[:, [false, true, false, false]] \ model.b
θ2 = [0, θ2...]
θ3 = [0, θ3..., 0]
θ4 = [0, θ4..., 0, 0]
diffusion_theta_history = [θ1 θ2 θ3 θ4]
reaction_theta_history = zeros(0, 4)
diffusion_vote_history = vote_history[1:end-1, :]
reaction_vote_history = zeros(0, 4)
regression_loss_history = [
    log(norm(model.A * θ1 - model.b)^2 / norm(model.b)^2 / size(model.A, 1)),
    log(norm(model.A * θ2 - model.b)^2 / norm(model.b)^2 / size(model.A, 1)),
    log(norm(model.A * θ3 - model.b)^2 / norm(model.b)^2 / size(model.A, 1)),
    log(norm(model.A * θ4 - model.b)^2 / norm(model.b)^2 / size(model.A, 1))
]
density_loss_history = [
    EQL.evaluate_density_loss(model, θ1, 2:length(sol), [true, true, true, true]),
    EQL.evaluate_density_loss(model, θ2, 2:length(sol), [true, true, true, true]),
    EQL.evaluate_density_loss(model, θ3, 2:length(sol), [true, true, true, true]),
    EQL.evaluate_density_loss(model, θ4, 2:length(sol), [true, true, true, true])
]
loss_history = 0 * regression_loss_history .+ density_loss_history .+ [4, 3, 2, 1]
pde = FVMProblem(prob, mesh_points; diffusion_function=diffusion_basis,
    diffusion_parameters=EQL.Parameters(p=diffusion_parameters, θ=θ4),
    proliferation=false)
pde_sol = solve(pde, TRBDF2(linsolve=KLUFactorization()), saveat=cell_sol.t)
diffusion_subset_history = indicator_history
reaction_subset_history = zeros(0, 4)
@test eql_sol.diffusion_theta == diffusion_theta
@test eql_sol.reaction_theta == reaction_theta
@test eql_sol.diffusion_subset == diffusion_subset
@test eql_sol.reaction_subset == reaction_subset
@test eql_sol.diffusion_theta_history == diffusion_theta_history
@test eql_sol.reaction_theta_history == reaction_theta_history
@test eql_sol.diffusion_vote_history == diffusion_vote_history
@test eql_sol.reaction_vote_history == reaction_vote_history
@test eql_sol.density_loss_history == density_loss_history
@test eql_sol.loss_history == loss_history
@test eql_sol.indicator_history == indicator_history
@test eql_sol.vote_history == vote_history
@test eql_sol.pde_sol.u ≈ pde_sol.u
@test diffusion_subset_history == eql_sol.diffusion_subset_history
@test reaction_subset_history == eql_sol.reaction_subset_history

# Test the show method 
eq = EQL.format_equation(eql_sol, :diffusion)
@test eq == "    D(q) = θ₂ᵈ ϕ₂ᵈ(q)"
stepwise_res, header_names = EQL.get_solution_table(eql_sol)
@test stepwise_res[:, 1] == [
    "θ₁ᵈ (votes)",
    "θ₂ᵈ (votes)",
    "θ₃ᵈ (votes)",
    "θ₄ᵈ (votes)",
    "Regression Loss",
    "Density Loss",
    "Loss"
]
@test stepwise_res[:, 2] == [
    @sprintf("%.2f", diffusion_theta_history[1, 1]) * " (" * @sprintf("%.2f", diffusion_vote_history[1, 1]) * ")",
    @sprintf("%.2f", diffusion_theta_history[2, 1]) * " (" * @sprintf("%.2f", diffusion_vote_history[2, 1]) * ")",
    @sprintf("%.2f", diffusion_theta_history[3, 1]) * " (" * @sprintf("%.2f", diffusion_vote_history[3, 1]) * ")",
    @sprintf("%.2f", diffusion_theta_history[4, 1]) * " (" * @sprintf("%.2f", diffusion_vote_history[4, 1]) * ")",
    @sprintf("%.2f", regression_loss_history[1]),
    @sprintf("%.2f", density_loss_history[1]),
    @sprintf("%.2f", loss_history[1])
]
@test stepwise_res[:, 3] == [
    @sprintf("%.2f", diffusion_theta_history[1, 2]) * " (" * @sprintf("%.2f", diffusion_vote_history[1, 2]) * ")",
    @sprintf("%.2f", diffusion_theta_history[2, 2]) * " (" * @sprintf("%.2f", diffusion_vote_history[2, 2]) * ")",
    @sprintf("%.2f", diffusion_theta_history[3, 2]) * " (" * @sprintf("%.2f", diffusion_vote_history[3, 2]) * ")",
    @sprintf("%.2f", diffusion_theta_history[4, 2]) * " (" * @sprintf("%.2f", diffusion_vote_history[4, 2]) * ")",
    @sprintf("%.2f", regression_loss_history[2]),
    @sprintf("%.2f", density_loss_history[2]),
    @sprintf("%.2f", loss_history[2])
]
@test stepwise_res[:, 4] == [
    @sprintf("%.2f", diffusion_theta_history[1, 3]) * " (" * @sprintf("%.2f", diffusion_vote_history[1, 3]) * ")",
    @sprintf("%.2f", diffusion_theta_history[2, 3]) * " (" * @sprintf("%.2f", diffusion_vote_history[2, 3]) * ")",
    @sprintf("%.2f", diffusion_theta_history[3, 3]) * " (" * @sprintf("%.2f", diffusion_vote_history[3, 3]) * ")",
    @sprintf("%.2f", diffusion_theta_history[4, 3]) * " (" * @sprintf("%.2f", diffusion_vote_history[4, 3]) * ")",
    @sprintf("%.2f", regression_loss_history[3]),
    @sprintf("%.2f", density_loss_history[3]),
    @sprintf("%.2f", loss_history[3])
]
@test stepwise_res[:, 5] == [
    @sprintf("%.2f", diffusion_theta_history[1, 4]) * " (" * @sprintf("%.2f", diffusion_vote_history[1, 4]) * ")",
    @sprintf("%.2f", diffusion_theta_history[2, 4]) * " (" * @sprintf("%.2f", diffusion_vote_history[2, 4]) * ")",
    @sprintf("%.2f", diffusion_theta_history[3, 4]) * " (" * @sprintf("%.2f", diffusion_vote_history[3, 4]) * ")",
    @sprintf("%.2f", diffusion_theta_history[4, 4]) * " (" * @sprintf("%.2f", diffusion_vote_history[4, 4]) * ")",
    @sprintf("%.2f", regression_loss_history[4]),
    @sprintf("%.2f", density_loss_history[4]),
    @sprintf("%.2f", loss_history[4])
]
@test header_names == [
    "Coefficient",
    "1",
    "2",
    "3",
    "4"
]

### Learning parameters: No cross-validation with indicator sampling 
cell_sol = sol
diffusion_basis = diffusion_basis
reaction_basis = BasisSet()
diffusion_parameters = nothing
reaction_parameters = nothing
diffusion_theta = nothing
reaction_theta = nothing
cross_validation = false
bidirectional = true
mesh_points = 250
threshold_tol = (q=1e-3, dt=1e-3)
density = true
regression = false
loss_function = default_loss(; density, regression)
trials = 100
skip = ()
initial = copy([true, true, true, true])
rng = Random.seed!(1234)
initial = :random
model_samples = 12
eql_sol = stepwise_selection(sol;
    diffusion_basis,
    reaction_basis,
    diffusion_parameters,
    reaction_parameters,
    diffusion_theta,
    reaction_theta,
    cross_validation,
    bidirectional,
    mesh_points,
    threshold_tol,
    density,
    regression,
    loss_function,
    trials,
    skip,
    initial,
    rng,
    model_samples
)

## Setting up the initial model 
pde = EQL.build_pde(cell_sol,
    mesh_points;
    diffusion_basis,
    reaction_basis,
    diffusion_parameters,
    reaction_parameters,
    diffusion_theta,
    reaction_theta)
model = EQL.EQLModel(cell_sol;
    diffusion_basis,
    reaction_basis,
    diffusion_parameters,
    reaction_parameters,
    diffusion_theta,
    reaction_theta,
    threshold_tol,
    mesh_points,
    pde)

## Simulating 
rng = Random.seed!(1234)
indicators = [EQL.get_indicators(model, initial, rng) for _ in 1:model_samples] |> unique!
sols = []
for indicator in indicators
    push!(sols,
        EQL.stepwise_selection(model, indicator;
            cross_validation,
            rng,
            skip,
            regression,
            density,
            loss_function,
            bidirectional,
            trials)
    )
end
ens_sol = EQL.EnsembleEQLSolution(sols)

## Test the field values 
final_loss = Dict()
average_loss = Dict()
for eql_sol in sols
    final_indicator = eql_sol.indicator_history[:, end]
    _final_loss = eql_sol.loss_history[end]
    if haskey(final_loss, final_indicator)
        final_loss[final_indicator] = (final_loss[final_indicator][1] + _final_loss,
            final_loss[final_indicator][2] + 1)
    else
        final_loss[final_indicator] = (_final_loss, 1)
    end
end
for (indicator, (loss, n)) in final_loss
    average_loss[indicator] = (loss / n, n)
end
@test ens_sol.final_loss == average_loss
@test ens_sol.best_model == [0, 1, 0, 0]
@test ens_sol.solutions == sols
for i in eachindex(sols)
    if i ∉ ens_sol.best_model_indices
        @test sols[i].indicator_history[:, end] ≠ [0, 1, 0, 0]
    else
        @test sols[i].indicator_history[:, end] == [0, 1, 0, 0]
    end
end
@test length(ens_sol.best_model_indices) == average_loss[[0, 1, 0, 0]][2]