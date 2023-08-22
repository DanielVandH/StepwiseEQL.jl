using ..StepwiseEQL
using EpithelialDynamics1D
using OrdinaryDiffEq
using LinearSolve
using StatsBase
using CairoMakie
using ReferenceTests
using FiniteVolumeMethod1D
const EQL = StepwiseEQL

final_time = 50.0
domain_length = 30.0
midpoint = domain_length / 2
cell_nodes = [LinRange(0, midpoint, 10); LinRange(midpoint, domain_length, 20)] |> unique!
damping_constant = 1.0
resting_spring_length = 1.0
spring_constant = 23.0
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
esol = solve(ens_prob, Tsit5(); trajectories=50, saveat=0.1)

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

stepwise_results = stepwise_learn(esol;
    diffusion_basis, diffusion_parameters,
    reaction_basis, reaction_parameters)
diff_infl = stack_diffusion_influence(stepwise_results)
react_infl = stack_reaction_influence(stepwise_results)

fig = Figure()
ax = Axis(fig[1, 1], xlabel=L"$ $Step", ylabel=L"$ $Influence",
    width=600, height=200,
    xticks=(1:length(stepwise_results), [L"%$s" for s in 1:length(stepwise_results)]))
all_diff_lines = Any[]
all_react_lines = Any[]
for i in axes(diff_infl, 1)
    line = stairs!(ax, eachindex(stepwise_results), diff_infl[i, :], linewidth=3)
    push!(all_diff_lines, line)
end
for i in axes(react_infl, 1)
    line = stairs!(ax, eachindex(stepwise_results), react_infl[i, :], linewidth=3, linestyle=:dash)
    push!(all_react_lines, line)
end
axislegend(ax,
    [all_diff_lines..., all_react_lines...],
    [L"\theta_1^d", L"\theta_2^d", L"\theta_3^d", L"\theta_1^r", L"\theta_2^r", L"\theta_3^r"],
    position=:rc)
resize_to_layout!(fig)
fig_path = normpath(@__DIR__, "..", "test", "figures")
@test_reference joinpath(fig_path, "reaction_eql_influence_aggregated.png") fig
fig

fvm_prob = FVMProblem(prob, 1000;
    diffusion_function=diffusion_basis,
    diffusion_parameters=Parameters(θ=stepwise_results[end].diffusion_θ, p=diffusion_parameters),
    reaction_function=reaction_basis,
    reaction_parameters=Parameters(θ=stepwise_results[end].reaction_θ, p=reaction_parameters))
fvm_sol = solve(fvm_prob, TRBDF2(linsolve=KLUFactorization()), saveat=esol[1].t)

fig = Figure(fontsize=41)
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"q(x, t)",
    width=600, height=300)
t_idx = floor.(Int64, LinRange(1, length(esol[1].t), 6))
colors = (:red, :blue, :black, :darkgreen, :orange, :purple)
q, r, q_means, q_lowers, q_uppers, q_knots = node_densities(esol)
for (j, i) in enumerate(t_idx)
    lines!(ax, fvm_prob.geometry.mesh_points, fvm_sol.u[i], color=colors[j], linewidth=5)
    band!(ax, q_knots[i], q_lowers[i], q_uppers[i], color=(colors[j], 0.3))
end
resize_to_layout!(fig)
@test_reference joinpath(fig_path, "reaction_eql_pde_aggregated.png") fig
fig
