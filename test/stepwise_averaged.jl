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

#### Averaged proliferation
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
mesh_points = 500
threshold_tol = (q=0.0, dt=0.0001)
density = true
regression = false
loss_function = default_loss(; density, regression)
trials = 20
skip = ()
initial = :all
aggregate = true
rng = Random.seed!(1234)
simulation_indices = rand(rng, eachindex(esol), 50)
num_knots = 1000
average = Val(true)
asol = EQL.AveragedODESolution(esol, num_knots, simulation_indices)
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
    rng,
    simulation_indices,
    num_knots,
    average)
A, b, idx_map, time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
    diffusion_basis, diffusion_parameters, diffusion_theta,
    reaction_basis, reaction_parameters, reaction_theta,
    threshold_tol,
    average,
    simulation_indices,
    num_knots)
@test asol.q == _sol.q
@test asol.u == _sol.u
@test asol.t == _sol.t
@test asol.cell_sol === _sol.cell_sol
@test eql_sol.model.A == A
@test eql_sol.model.b == b
@test eql_sol.model.idx_map == idx_map
@test eql_sol.model.valid_time_indices == time_indices