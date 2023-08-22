using ..StepwiseEQL
using EpithelialDynamics1D
using OrdinaryDiffEq
using LinearSolve
using LinearAlgebra
using StatsBase
using CairoMakie
using FiniteVolumeMethod1D
using Random
using StableRNGs
using Printf
using ElasticArrays
using DataInterpolations
const EQL = StepwiseEQL

#### Aggregated proliferation
### Setup
final_time = 50.0
domain_length = 30.0
midpoint = domain_length / 2
cell_nodes = [LinRange(0, midpoint, 10); LinRange(midpoint, domain_length, 20)] |> unique!
damping_constant = 1.0
resting_spring_length = 1.0
spring_constant = 50.0
k = spring_constant
force_law_parameters = (s=resting_spring_length, k=spring_constant)
force_law = (δ, p) -> p.k * (p.s - δ)
Δt = 1e-2
K = 15.0
β = 1e-2
G = (δ, p) -> max(zero(δ), p.β * p.K * (one(δ) - inv(p.K * δ)))
Gp = (β=β, K=K)
prob = CellProblem(;
    final_time,
    initial_condition=cell_nodes,
    damping_constant,
    force_law,
    force_law_parameters,
    proliferation_law=G,
    proliferation_period=Δt,
    proliferation_law_parameters=Gp)
ens_prob = EnsembleProblem(prob)
Random.seed!(191919)
esol = solve(ens_prob, Tsit5(), EnsembleSerial(); trajectories=15, saveat=0.1)

### Learning parameters: No cross-validation and no indicator sampling
diffusion_basis = BasisSet(
    (u, k) -> k * inv(u),
    (u, k) -> k * inv(u^2),
    (u, k) -> k * inv(u^3)
)
diffusion_parameters = k
reaction_basis = BasisSet(
    (u, (β, K)) -> β * K * u,
    (u, (β, K)) -> β * u^2,
    (u, (β, K)) -> β * u^3
)
reaction_parameters = Gp
cell_sol = esol
diffusion_theta = nothing
reaction_theta = nothing
cross_validation = false
bidirectional = true
mesh_points = 250
threshold_tol = (q=1e-2, dt=1e-2)
density = true
regression = false
loss_function = default_loss(; density, regression)
trials = 20
skip = ()
initial = copy([false, true, false, true, true, false])
aggregate = true
rng = Random.seed!(1234)
@time eql_sol = stepwise_selection(cell_sol;
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
@test !model_changed
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
indicators = [false, false, false, true, true, false]
θ = model.A[:, indicators] \ model.b
diffusion_theta = [0.0, 0.0, 0.0]
reaction_theta = [θ..., 0.0]
diffusion_subset = Int[]
reaction_subset = [1, 2]
θ1 = model.A[:, [false, true, false, true, true, false]] \ model.b
θ2 = model.A[:, [false, false, false, true, true, false]] \ model.b
θ1 = [0.0, θ1[1], 0.0, θ1[2], θ1[3], 0.0]
θ2 = [0.0, 0.0, 0.0, θ2[1], θ2[2], 0.0]
diffusion_theta_history = [θ1[1:3] zeros(3)]
reaction_theta_history = [[θ1[4:5]; 0.0] [θ2[4:5]; 0.0]]
diffusion_vote_history = vote_history[1:3, :]
reaction_vote_history = vote_history[4:6, :]
regression_loss_history = [
    log(norm(model.A * θ1 - model.b)^2 / norm(model.b)^2 / size(model.A, 1)),
    log(norm(model.A * θ2 - model.b)^2 / norm(model.b)^2 / size(model.A, 1)),
]
density_loss_history = [
    EQL.evaluate_density_loss(model, θ1, model.valid_time_indices, [false, true, false, true, true, false]),
    EQL.evaluate_density_loss(model, θ2, model.valid_time_indices, [false, false, false, true, true, false])
]
loss_history = 0 * regression_loss_history .+ density_loss_history .+ [3, 2]
pde = FVMProblem(prob, mesh_points; diffusion_function=diffusion_basis,
    diffusion_parameters=EQL.Parameters(p=diffusion_parameters, θ=zeros(3)),
    reaction_function=reaction_basis,
    reaction_parameters=EQL.Parameters(p=reaction_parameters, θ=[θ2[4], θ2[5], 0.0]),
    proliferation=true)
pde_sol = solve(pde, TRBDF2(linsolve=KLUFactorization()), saveat=cell_sol[1].t)
diffusion_subset_history = indicator_history[1:3, :]
reaction_subset_history = indicator_history[4:6, :]
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
@test eq == "    D(q) = 0"
eq = EQL.format_equation(eql_sol, :reaction)
@test eq == "    R(q) = θ₁ʳ ϕ₁ʳ(q) + θ₂ʳ ϕ₂ʳ(q)"
stepwise_res, header_names = EQL.get_solution_table(eql_sol)
@test stepwise_res[:, 1] == [
    "θ₁ᵈ (votes)",
    "θ₂ᵈ (votes)",
    "θ₃ᵈ (votes)",
    "θ₁ʳ (votes)",
    "θ₂ʳ (votes)",
    "θ₃ʳ (votes)",
    "Regression Loss",
    "Density Loss",
    "Loss"
]
@test stepwise_res[:, 2] == [
    @sprintf("%.2f", diffusion_theta_history[1, 1]) * " (" * @sprintf("%.2f", diffusion_vote_history[1, 1]) * ")",
    @sprintf("%.2f", diffusion_theta_history[2, 1]) * " (" * @sprintf("%.2f", diffusion_vote_history[2, 1]) * ")",
    @sprintf("%.2f", diffusion_theta_history[3, 1]) * " (" * @sprintf("%.2f", diffusion_vote_history[3, 1]) * ")",
    @sprintf("%.2f", reaction_theta_history[1, 1]) * " (" * @sprintf("%.2f", reaction_vote_history[1, 1]) * ")",
    @sprintf("%.2f", reaction_theta_history[2, 1]) * " (" * @sprintf("%.2f", reaction_vote_history[2, 1]) * ")",
    @sprintf("%.2f", reaction_theta_history[3, 1]) * " (" * @sprintf("%.2f", reaction_vote_history[3, 1]) * ")",
    @sprintf("%.2f", regression_loss_history[1]),
    @sprintf("%.2f", density_loss_history[1]),
    @sprintf("%.2f", loss_history[1])
]
@test stepwise_res[:, 3] == [
    @sprintf("%.2f", diffusion_theta_history[1, 2]) * " (" * @sprintf("%.2f", diffusion_vote_history[1, 2]) * ")",
    @sprintf("%.2f", diffusion_theta_history[2, 2]) * " (" * @sprintf("%.2f", diffusion_vote_history[2, 2]) * ")",
    @sprintf("%.2f", diffusion_theta_history[3, 2]) * " (" * @sprintf("%.2f", diffusion_vote_history[3, 2]) * ")",
    @sprintf("%.2f", reaction_theta_history[1, 2]) * " (" * @sprintf("%.2f", reaction_vote_history[1, 2]) * ")",
    @sprintf("%.2f", reaction_theta_history[2, 2]) * " (" * @sprintf("%.2f", reaction_vote_history[2, 2]) * ")",
    @sprintf("%.2f", reaction_theta_history[3, 2]) * " (" * @sprintf("%.2f", reaction_vote_history[3, 2]) * ")",
    @sprintf("%.2f", regression_loss_history[2]),
    @sprintf("%.2f", density_loss_history[2]),
    @sprintf("%.2f", loss_history[2])
]
@test header_names == [
    "Coefficient",
    "1",
    "2"
]

### Learning parameters: No cross-validation with indicator sampling 
initial = :random
model_samples = 12
rng = Random.seed!(1234)
@time eql_sol = stepwise_selection(cell_sol;
    diffusion_basis,
    reaction_basis,
    diffusion_parameters,
    reaction_parameters,
    diffusion_theta=nothing,
    reaction_theta=nothing,
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
    diffusion_theta=nothing,
    reaction_theta=nothing)
model = EQL.EQLModel(cell_sol;
    diffusion_basis,
    reaction_basis,
    diffusion_parameters,
    reaction_parameters,
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
@test ens_sol.final_loss == eql_sol.final_loss == average_loss
@test ens_sol.best_model == eql_sol.best_model == [0, 0, 0, 1, 1, 0]
@test ens_sol.solutions == sols
for i in eachindex(sols)
    if i ∉ ens_sol.best_model_indices
        @test sols[i].indicator_history[:, end] ≠ [0, 0, 0, 1, 1, 0]
    else
        @test sols[i].indicator_history[:, end] == [0, 0, 0, 1, 1, 0]
    end
end
@test length(ens_sol.best_model_indices) == length(eql_sol.best_model_indices) == average_loss[[0, 0, 0, 1, 1, 0]][2]

## Test the show method
@testset "Testing a specific proliferation example to check loss function for an aggregated problem" begin
    final_time = 50.0
    domain_length = 30.0
    midpoint = domain_length / 2
    cell_nodes = [LinRange(0, midpoint, 10); LinRange(midpoint, domain_length, 20)] |> unique!
    damping_constant = 1.0
    resting_spring_length = 1.0
    spring_constant = 50.0
    k = spring_constant
    force_law_parameters = (s=resting_spring_length, k=spring_constant)
    force_law = (δ, p) -> p.k * (p.s - δ)
    Δt = 1e-2
    K = 15.0
    β = 1e-2
    G = (δ, p) -> max(zero(δ), p.β * p.K * (one(δ) - inv(p.K * δ)))
    Gp = (β=β, K=K)
    prob = CellProblem(;
        final_time,
        initial_condition=cell_nodes,
        damping_constant,
        force_law,
        force_law_parameters,
        proliferation_law=G,
        proliferation_period=Δt,
        proliferation_law_parameters=Gp)
    rng = StableRNG(123)
    ens_prob = EnsembleProblem(prob, rng=rng)
    diffusion_basis = BasisSet(
        (u, _) -> inv(u),
        (u, _) -> inv(u^2),
        (u, _) -> inv(u^3)
    )
    reaction_basis = BasisSet(
        (u, _) -> u,
        (u, _) -> u^2,
        (u, _) -> u^3,
        (u, _) -> u^4
    )
    esol = solve(ens_prob, Tsit5(), EnsembleSerial(), trajectories=50, saveat=0.1)
    eql_sol = stepwise_selection(esol; average=Val(false), aggregate=true,
        diffusion_basis, reaction_basis,
        diffusion_theta=[0.0, spring_constant / damping_constant, 0.0],
        density=true, regression=false, cross_validation=false)
    fvm_prob = FVMProblem(
        prob;
        diffusion_parameters=EQL.Parameters(θ=[0.0, spring_constant / damping_constant, 0.0], p=nothing),
        diffusion_function=diffusion_basis,
        reaction_parameters=EQL.Parameters(θ=eql_sol.reaction_theta, p=nothing),
        reaction_function=reaction_basis,
        proliferation=true)
    fvm_sol = solve(fvm_prob, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    penalty = count(!iszero, eql_sol.reaction_theta)
    pde_loss = 0.0
    ctr = 1
    for k in 1:50
        for j in 2:length(0:0.1:50)
            for i in 1:length(esol[k].u[j])
                cell_q = node_densities(esol[k].u[j])[i]
                fvm_q = LinearInterpolation(fvm_sol.u[j], fvm_sol.prob.p.geometry.mesh_points)(esol[k].u[j][i])
                pde_loss += (fvm_q - cell_q)^2 / cell_q^2
                ctr += 1
            end
        end
    end
    @test eql_sol.loss_history[end] ≈ penalty + log(pde_loss / ctr) rtol = 1e-3

    ## Also test the matrices 
    n = 0
    for k in eachindex(esol)
        for j in (firstindex(esol[k])+1):lastindex(esol[k])
            for i in eachindex(esol[k].u[j])
                n += 1
            end
        end
    end
    q = zeros(n)
    ∂q = zeros(n)
    ∂²q = zeros(n)
    ∂qt = zeros(n)
    ctr = 1
    for k in eachindex(esol)
        for j in (firstindex(esol[k])+1):lastindex(esol[k])
            for i in eachindex(esol[k].u[j])
                _q = EQL.cell_density(esol[k], i, j)
                _∂q = EQL.cell_∂q∂x(esol[k], i, j)
                _∂²q = EQL.cell_∂²q∂x²(esol[k], i, j)
                _∂qt = EQL.cell_∂q∂t(esol[k], i, j)
                if all(≥(0), abs.((_∂q, _∂²q, _∂qt)))
                    q[ctr] = _q
                    ∂q[ctr] = _∂q
                    ∂²q[ctr] = _∂²q
                    ∂qt[ctr] = _∂qt
                    ctr += 1
                end
            end
        end
    end
    A = [q q .^ 2 q .^ 3 q .^ 4]
    b = ∂qt .- 50 ./ q .^ 2 .* ∂²q .+ 50 .* 2 ./ q .^ 3 .* ∂q .^ 2
    @test A ≈ eql_sol.model.A
    @test b ≈ eql_sol.model.b
end