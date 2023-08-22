```@meta
EditURL = "https://github.com/DanielVandH/StepwiseEQL.jl/tree/main/docs/src/case_studies/cs4b.md"
```

!!! tip
    This example is also available as a Jupyter notebook:
    [`cs4b.ipynb`](https://github.com/DanielVandH/StepwiseEQL.jl/tree/main/docs/src/case_studies/notebooks/cs4b.ipynb)

# Case Studies: Case Study 4; Inaccurate continuum limit

This example shows how we obtained the results in the paper
for the fourth case study, for the case that the continuum limit is
inaccurate. Let us load in the packages we will need.

```julia
using StepwiseEQL
using CairoMakie
using EpithelialDynamics1D
using OrdinaryDiffEq
using Random
```

## Simulating
Let us start by simulating the problem. We use the force law $F(\ell) = k(s-\ell)$
and the proliferation law $G(\ell) = \beta[1 - 1/(K\ell)]$.

```julia
final_time = 250.0
domain_length = 30.0
midpoint = domain_length / 2
initial_condition = [LinRange(0, 5, 30);] |> unique!
damping_constant = 1.0
resting_spring_length = 0.2
spring_constant = 1 / 5
k = spring_constant
η = damping_constant
s = resting_spring_length
force_law_parameters = (s=resting_spring_length, k=spring_constant)
force_law = (δ, p) -> p.k * (p.s - δ)
Δt = 1e-2
K = 15.0
β = 0.15
G = (δ, p) -> p.β * (one(δ) - inv(p.K * δ))
Gp = (β=β, K=K)
prob = CellProblem(;
    final_time,
    initial_condition,
    damping_constant,
    force_law,
    force_law_parameters,
    proliferation_law=G,
    proliferation_period=Δt,
    proliferation_law_parameters=Gp,
    fix_right=false)
ens_prob = EnsembleProblem(prob)
Random.seed!(292919)
interval_1 = (0.0, 2, 20)
interval_2 = (2, 10, 200)
interval_3 = (10.0, 20, 200)
interval_4 = (20, 50, 200)
t = [0, 5, 25, 50, 100, 250]
saveat = [t
             LinRange(interval_1...)
             LinRange(interval_2...)
             LinRange(interval_3...)
             LinRange(interval_4...)] |> unique! |> sort!
esol = solve(ens_prob, Tsit5(), EnsembleSerial(); trajectories=1000, saveat=saveat)
```

## Equation learning
We now learn the equations, applying our sequential procedure.

```julia
diffusion_basis = PolynomialBasis(-1, -3)
reaction_basis = PolynomialBasis(1, 5)
rhs_basis = PolynomialBasis(1, 5)
moving_boundary_basis = PolynomialBasis(-1, -3)
```
```
(::BasisSet{Tuple{StepwiseEQL.var"#52#54"{Int64}, StepwiseEQL.var"#52#54"{Int64}, StepwiseEQL.var"#52#54"{Int64}}}) (generic function with 3 methods)
```

```julia
eql_sol = stepwise_selection(esol; diffusion_basis, reaction_basis,
    rhs_basis, moving_boundary_basis,
    reaction_theta=zeros(5), moving_boundary_theta=zeros(3), rhs_theta=zeros(5),
    mesh_points=100,
    num_knots=50, threshold_tol=(dt=0.4,),
    initial=:none, time_range=(interval_1[1], interval_1[2]))
```
```
StepwiseEQL Solution.
    D(q) = θ₂ᵈ ϕ₂ᵈ(q)
┌──────┬──────────────────┬────────┐
│ Step │  θ₁ᵈ   θ₂ᵈ   θ₃ᵈ │   Loss │
├──────┼──────────────────┼────────┤
│    1 │ 0.00  0.00  0.00 │ -16.64 │
│    2 │ 0.00  0.21  0.00 │ -15.64 │
└──────┴──────────────────┴────────┘
```

```julia
eql_sol_2 = stepwise_selection(esol; diffusion_basis, reaction_basis,
    rhs_basis, moving_boundary_basis,
    diffusion_theta=eql_sol.diffusion_theta,
    reaction_theta=zeros(5), rhs_theta=zeros(5),
    mesh_points=100, threshold_tol=(dL=0.4,),
    num_knots=100, initial=:none,
    time_range=(interval_2[1], interval_2[2]))
```
```
┌──────┬──────────────────┬────────┐
│ Step │  θ₁ᵉ   θ₂ᵉ   θ₃ᵉ │   Loss │
├──────┼──────────────────┼────────┤
│    1 │ 0.00  0.00  0.00 │ -11.24 │
│    2 │ 0.00  0.23  0.00 │ -10.24 │
└──────┴──────────────────┴────────┘
```

```julia
eql_sol_3 = stepwise_selection(esol; diffusion_basis, reaction_basis,
    rhs_basis, moving_boundary_basis,
    diffusion_theta=eql_sol.diffusion_theta, reaction_theta=zeros(5),
    moving_boundary_theta=eql_sol_2.moving_boundary_theta,
    mesh_points=100, num_knots=100,
    initial=:none,
    time_range=(interval_3[1], interval_3[2]))
```
```
StepwiseEQL Solution.
    H(q) = θ₁ʰ ϕ₁ʰ(q) + θ₄ʰ ϕ₄ʰ(q)
┌──────┬────────────────────────────────┬────────┐
│ Step │   θ₁ʰ   θ₂ʰ   θ₃ʰ    θ₄ʰ   θ₅ʰ │   Loss │
├──────┼────────────────────────────────┼────────┤
│    1 │  0.00  0.00  0.00   0.00  0.00 │  -8.31 │
│    2 │  0.00  0.00  0.00  -0.01  0.00 │ -11.25 │
│    3 │ -0.15  0.00  0.00  -0.01  0.00 │ -12.31 │
└──────┴────────────────────────────────┴────────┘
```

```julia
eql_sol_4 = stepwise_selection(esol; diffusion_basis, reaction_basis,
    rhs_basis, moving_boundary_basis,
    diffusion_theta=eql_sol.diffusion_theta,
    moving_boundary_theta=eql_sol_2.moving_boundary_theta,
    rhs_theta=eql_sol_3.rhs_theta, mesh_points=1000,
    num_knots=100,
    threshold_tol=(q=0.3,),
    initial=:none, time_range=(interval_4[1], interval_4[2]))
```
```
StepwiseEQL Solution.
    R(q) = θ₁ʳ ϕ₁ʳ(q) + θ₂ʳ ϕ₂ʳ(q)
┌──────┬───────────────────────────────┬────────┐
│ Step │  θ₁ʳ    θ₂ʳ   θ₃ʳ   θ₄ʳ   θ₅ʳ │   Loss │
├──────┼───────────────────────────────┼────────┤
│    1 │ 0.00   0.00  0.00  0.00  0.00 │  -9.40 │
│    2 │ 0.00   0.00  0.00  0.00  0.00 │  -8.70 │
│    3 │ 0.11  -0.01  0.00  0.00  0.00 │ -17.74 │
└──────┴───────────────────────────────┴────────┘
```

## Plotting
Now we plot our results.

```julia
fig = Figure(fontsize=81, resolution=(3250, 2070))
ax_pde = Axis(fig[1, 1],
    xlabel=L"x", ylabel=L"q(x, t)",
    title=L"(a):$ $ PDE comparison",
    titlealign=:left,
    width=1200, height=400,
    xticks=(0:5:40, [L"%$s" for s in 0:5:40]),
    yticks=(0:5:15, [L"%$s" for s in 0:5:15])
)
xlims!(ax_pde, 0, 10)
ylims!(ax_pde, 0, 20)
ax_leading_edge = Axis(fig[1, 2],
    xlabel=L"t", ylabel=L"L(t)",
    title=L"(b):$ $ Leading edge comparison",
    titlealign=:left,
    width=1200, height=400,
    xticks=(0:50:250, [L"%$s" for s in 0:50:250]),
    yticks=(0:5:10, [L"%$s" for s in 0:5:10])
)
xlims!(ax_leading_edge, 0, 250)
ylims!(ax_leading_edge, 0, 10)
ax_dq = Axis(fig[2, 1],
    xlabel=L"q", ylabel=L"D(q)",
    title=L"(c): $D(q)$ comparison",
    titlealign=:left,
    width=1200, height=400,
    xticks=(5:5:15, [L"%$s" for s in 5:5:15]),
    yticks=(0.0005:0.0025:0.01, [L"%$s" for s in 0.0005:0.0025:0.01])
)
q_min, q_max = extrema(stack(eql_sol_4.model.cell_sol.q))
q_min = 5
xlims!(ax_dq, q_min, q_max)
ylims!(ax_dq, 0.0005, 0.01)
ax_rq = Axis(fig[2, 2],
    xlabel=L"q", ylabel=L"R(q)",
    title=L"(d): $R(q)$ comparison",
    titlealign=:left,
    width=1200, height=400,
    xticks=(5:5:15, [L"%$s" for s in 5:5:15]),
    yticks=(-0.25:0.25:0.75, [L"%$s" for s in -0.25:0.25:0.75])
)
xlims!(ax_rq, q_min, q_max)
ylims!(ax_rq, -0.25, 0.75)
ax_hq = Axis(fig[3, 1],
    xlabel=L"q", ylabel=L"H(q)",
    title=L"(e): $H(q)$ comparison",
    titlealign=:left,
    width=1200, height=400,
    xticks=(5:5:15, [L"%$s" for s in 5:5:15]),
    yticks=(-24:6:0, [L"%$s" for s in -24:6:0])
)
xlims!(ax_hq, q_min, q_max)
ylims!(ax_hq, -24, 2)
ax_eq = Axis(fig[3, 2],
    xlabel=L"q", ylabel=L"E(q)",
    title=L"(f): $E(q)$ comparison",
    titlealign=:left,
    width=1200, height=400,
    xticks=(5:5:15, [L"%$s" for s in 5:5:15]),
    yticks=(0.0005:0.0025:0.01, [L"%$s" for s in 0.0005:0.0025:0.01])
)
xlims!(ax_eq, q_min, q_max)
ylims!(ax_eq, 0.0005, 0.01)
time_indices = [findlast(≤(τ), esol[1].t) for τ in t]
colors = (:black, :red, :blue, :green, :orange, :purple, :brown)
pde_ξ = eql_sol_4.pde.geometry.mesh_points
pde_L = eql_sol_4.pde_sol[end, :]
pde_q = eql_sol_4.pde_sol[begin:(end-1), :]
cell_q = eql_sol_4.model.cell_sol.q
cell_r = eql_sol_4.model.cell_sol.u
cell_L = last.(eql_sol_4.model.cell_sol.u)

q_range = LinRange(q_min, q_max, 100)
D_cont_fnc = q -> (k / η) / q^2
R_cont_fnc = q -> β * q * (1 - q / K)
H_cont_fnc = q -> 2q^2 * (1 - q * s)
E_cont_fnc = D_cont_fnc
D_cont = D_cont_fnc.(q_range)
R_cont = R_cont_fnc.(q_range)
H_cont = H_cont_fnc.(q_range)
E_cont = E_cont_fnc.(q_range)
D_sol = diffusion_basis.(q_range, Ref(eql_sol.diffusion_theta), Ref(nothing))
R_sol = reaction_basis.(q_range, Ref(eql_sol_4.reaction_theta), Ref(nothing))
H_sol = rhs_basis.(q_range, Ref(eql_sol_3.rhs_theta), Ref(nothing))
E_sol = moving_boundary_basis.(q_range, Ref(eql_sol_2.moving_boundary_theta), Ref(nothing))

for (j, i) in enumerate(time_indices)
    lines!(ax_pde, pde_ξ * pde_L[i], pde_q[:, i], color=colors[j], linestyle=:dash, linewidth=8)
    lines!(ax_pde, cell_r[i], cell_q[i], color=colors[j], linewidth=4, label=L"%$(t[j])")
end
arrows!(ax_pde, [5.0], [3.0], [3.0], [0.0], color=:black, linewidth=12, arrowsize=57)
text!(ax_pde, [7.0], [3.5], text=L"t", fontsize=81)

lines!(ax_leading_edge, esol[1].t, pde_L, color=:red, linestyle=:dash, linewidth=5, label=L"$ $Learned")
lines!(ax_leading_edge, esol[1].t, cell_L, color=:black, linewidth=3, label=L"$ $Discrete")
axislegend(ax_leading_edge, position=:rb)

lines!(ax_dq, q_range, D_sol, linewidth=6, color=:red, linestyle=:solid, label=L"$ $Learned")
lines!(ax_dq, q_range, D_cont, linewidth=6, color=:black, linestyle=:dash, label=L"$ $Continuum limit")
axislegend(ax_dq, position=:rt)

lines!(ax_rq, q_range, R_sol, linewidth=6, color=:red, linestyle=:solid, label=L"$ $Learned")
lines!(ax_rq, q_range, R_cont, linewidth=6, color=:black, linestyle=:dash, label=L"$ $Continuum limit")
hlines!(ax_rq, [0.0], color=:grey, linewidth=12, linestyle=:dash)
axislegend(ax_rq, position=:lb)

lines!(ax_hq, q_range, H_sol, linewidth=6, color=:red, linestyle=:solid, label=L"$ $Learned")
lines!(ax_hq, q_range, H_cont, linewidth=6, color=:black, linestyle=:dash, label=L"$ $Continuum limit")
axislegend(ax_hq, position=:rt)

lines!(ax_eq, q_range, E_sol, linewidth=6, color=:red, linestyle=:solid, label=L"$ $Learned")
lines!(ax_eq, q_range, E_cont, linewidth=6, color=:black, linestyle=:dash, label=L"$ $Continuum limit")
axislegend(ax_eq, position=:rt)
fig
```
```@raw html
<figure>
    <img src='../../figures/figure11.png', alt='Figure 11 from the paper'><br>
</figure>
```

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/DanielVandH/StepwiseEQL.jl/tree/main/docs/src/literate_case_studies/cs4b.jl).

```julia
using StepwiseEQL
using CairoMakie
using EpithelialDynamics1D
using OrdinaryDiffEq
using Random

final_time = 250.0
domain_length = 30.0
midpoint = domain_length / 2
initial_condition = [LinRange(0, 5, 30);] |> unique!
damping_constant = 1.0
resting_spring_length = 0.2
spring_constant = 1 / 5
k = spring_constant
η = damping_constant
s = resting_spring_length
force_law_parameters = (s=resting_spring_length, k=spring_constant)
force_law = (δ, p) -> p.k * (p.s - δ)
Δt = 1e-2
K = 15.0
β = 0.15
G = (δ, p) -> p.β * (one(δ) - inv(p.K * δ))
Gp = (β=β, K=K)
prob = CellProblem(;
    final_time,
    initial_condition,
    damping_constant,
    force_law,
    force_law_parameters,
    proliferation_law=G,
    proliferation_period=Δt,
    proliferation_law_parameters=Gp,
    fix_right=false)
ens_prob = EnsembleProblem(prob)
Random.seed!(292919)
interval_1 = (0.0, 2, 20)
interval_2 = (2, 10, 200)
interval_3 = (10.0, 20, 200)
interval_4 = (20, 50, 200)
t = [0, 5, 25, 50, 100, 250]
saveat = [t
             LinRange(interval_1...)
             LinRange(interval_2...)
             LinRange(interval_3...)
             LinRange(interval_4...)] |> unique! |> sort!
esol = solve(ens_prob, Tsit5(), EnsembleSerial(); trajectories=1000, saveat=saveat)

diffusion_basis = PolynomialBasis(-1, -3)
reaction_basis = PolynomialBasis(1, 5)
rhs_basis = PolynomialBasis(1, 5)
moving_boundary_basis = PolynomialBasis(-1, -3)

eql_sol = stepwise_selection(esol; diffusion_basis, reaction_basis,
    rhs_basis, moving_boundary_basis,
    reaction_theta=zeros(5), moving_boundary_theta=zeros(3), rhs_theta=zeros(5),
    mesh_points=100,
    num_knots=50, threshold_tol=(dt=0.4,),
    initial=:none, time_range=(interval_1[1], interval_1[2]))

eql_sol_2 = stepwise_selection(esol; diffusion_basis, reaction_basis,
    rhs_basis, moving_boundary_basis,
    diffusion_theta=eql_sol.diffusion_theta,
    reaction_theta=zeros(5), rhs_theta=zeros(5),
    mesh_points=100, threshold_tol=(dL=0.4,),
    num_knots=100, initial=:none,
    time_range=(interval_2[1], interval_2[2]))

eql_sol_3 = stepwise_selection(esol; diffusion_basis, reaction_basis,
    rhs_basis, moving_boundary_basis,
    diffusion_theta=eql_sol.diffusion_theta, reaction_theta=zeros(5),
    moving_boundary_theta=eql_sol_2.moving_boundary_theta,
    mesh_points=100, num_knots=100,
    initial=:none,
    time_range=(interval_3[1], interval_3[2]))

eql_sol_4 = stepwise_selection(esol; diffusion_basis, reaction_basis,
    rhs_basis, moving_boundary_basis,
    diffusion_theta=eql_sol.diffusion_theta,
    moving_boundary_theta=eql_sol_2.moving_boundary_theta,
    rhs_theta=eql_sol_3.rhs_theta, mesh_points=1000,
    num_knots=100,
    threshold_tol=(q=0.3,),
    initial=:none, time_range=(interval_4[1], interval_4[2]))

fig = Figure(fontsize=81, resolution=(3250, 2070))
ax_pde = Axis(fig[1, 1],
    xlabel=L"x", ylabel=L"q(x, t)",
    title=L"(a):$ $ PDE comparison",
    titlealign=:left,
    width=1200, height=400,
    xticks=(0:5:40, [L"%$s" for s in 0:5:40]),
    yticks=(0:5:15, [L"%$s" for s in 0:5:15])
)
xlims!(ax_pde, 0, 10)
ylims!(ax_pde, 0, 20)
ax_leading_edge = Axis(fig[1, 2],
    xlabel=L"t", ylabel=L"L(t)",
    title=L"(b):$ $ Leading edge comparison",
    titlealign=:left,
    width=1200, height=400,
    xticks=(0:50:250, [L"%$s" for s in 0:50:250]),
    yticks=(0:5:10, [L"%$s" for s in 0:5:10])
)
xlims!(ax_leading_edge, 0, 250)
ylims!(ax_leading_edge, 0, 10)
ax_dq = Axis(fig[2, 1],
    xlabel=L"q", ylabel=L"D(q)",
    title=L"(c): $D(q)$ comparison",
    titlealign=:left,
    width=1200, height=400,
    xticks=(5:5:15, [L"%$s" for s in 5:5:15]),
    yticks=(0.0005:0.0025:0.01, [L"%$s" for s in 0.0005:0.0025:0.01])
)
q_min, q_max = extrema(stack(eql_sol_4.model.cell_sol.q))
q_min = 5
xlims!(ax_dq, q_min, q_max)
ylims!(ax_dq, 0.0005, 0.01)
ax_rq = Axis(fig[2, 2],
    xlabel=L"q", ylabel=L"R(q)",
    title=L"(d): $R(q)$ comparison",
    titlealign=:left,
    width=1200, height=400,
    xticks=(5:5:15, [L"%$s" for s in 5:5:15]),
    yticks=(-0.25:0.25:0.75, [L"%$s" for s in -0.25:0.25:0.75])
)
xlims!(ax_rq, q_min, q_max)
ylims!(ax_rq, -0.25, 0.75)
ax_hq = Axis(fig[3, 1],
    xlabel=L"q", ylabel=L"H(q)",
    title=L"(e): $H(q)$ comparison",
    titlealign=:left,
    width=1200, height=400,
    xticks=(5:5:15, [L"%$s" for s in 5:5:15]),
    yticks=(-24:6:0, [L"%$s" for s in -24:6:0])
)
xlims!(ax_hq, q_min, q_max)
ylims!(ax_hq, -24, 2)
ax_eq = Axis(fig[3, 2],
    xlabel=L"q", ylabel=L"E(q)",
    title=L"(f): $E(q)$ comparison",
    titlealign=:left,
    width=1200, height=400,
    xticks=(5:5:15, [L"%$s" for s in 5:5:15]),
    yticks=(0.0005:0.0025:0.01, [L"%$s" for s in 0.0005:0.0025:0.01])
)
xlims!(ax_eq, q_min, q_max)
ylims!(ax_eq, 0.0005, 0.01)
time_indices = [findlast(≤(τ), esol[1].t) for τ in t]
colors = (:black, :red, :blue, :green, :orange, :purple, :brown)
pde_ξ = eql_sol_4.pde.geometry.mesh_points
pde_L = eql_sol_4.pde_sol[end, :]
pde_q = eql_sol_4.pde_sol[begin:(end-1), :]
cell_q = eql_sol_4.model.cell_sol.q
cell_r = eql_sol_4.model.cell_sol.u
cell_L = last.(eql_sol_4.model.cell_sol.u)

q_range = LinRange(q_min, q_max, 100)
D_cont_fnc = q -> (k / η) / q^2
R_cont_fnc = q -> β * q * (1 - q / K)
H_cont_fnc = q -> 2q^2 * (1 - q * s)
E_cont_fnc = D_cont_fnc
D_cont = D_cont_fnc.(q_range)
R_cont = R_cont_fnc.(q_range)
H_cont = H_cont_fnc.(q_range)
E_cont = E_cont_fnc.(q_range)
D_sol = diffusion_basis.(q_range, Ref(eql_sol.diffusion_theta), Ref(nothing))
R_sol = reaction_basis.(q_range, Ref(eql_sol_4.reaction_theta), Ref(nothing))
H_sol = rhs_basis.(q_range, Ref(eql_sol_3.rhs_theta), Ref(nothing))
E_sol = moving_boundary_basis.(q_range, Ref(eql_sol_2.moving_boundary_theta), Ref(nothing))

for (j, i) in enumerate(time_indices)
    lines!(ax_pde, pde_ξ * pde_L[i], pde_q[:, i], color=colors[j], linestyle=:dash, linewidth=8)
    lines!(ax_pde, cell_r[i], cell_q[i], color=colors[j], linewidth=4, label=L"%$(t[j])")
end
arrows!(ax_pde, [5.0], [3.0], [3.0], [0.0], color=:black, linewidth=12, arrowsize=57)
text!(ax_pde, [7.0], [3.5], text=L"t", fontsize=81)

lines!(ax_leading_edge, esol[1].t, pde_L, color=:red, linestyle=:dash, linewidth=5, label=L"$ $Learned")
lines!(ax_leading_edge, esol[1].t, cell_L, color=:black, linewidth=3, label=L"$ $Discrete")
axislegend(ax_leading_edge, position=:rb)

lines!(ax_dq, q_range, D_sol, linewidth=6, color=:red, linestyle=:solid, label=L"$ $Learned")
lines!(ax_dq, q_range, D_cont, linewidth=6, color=:black, linestyle=:dash, label=L"$ $Continuum limit")
axislegend(ax_dq, position=:rt)

lines!(ax_rq, q_range, R_sol, linewidth=6, color=:red, linestyle=:solid, label=L"$ $Learned")
lines!(ax_rq, q_range, R_cont, linewidth=6, color=:black, linestyle=:dash, label=L"$ $Continuum limit")
hlines!(ax_rq, [0.0], color=:grey, linewidth=12, linestyle=:dash)
axislegend(ax_rq, position=:lb)

lines!(ax_hq, q_range, H_sol, linewidth=6, color=:red, linestyle=:solid, label=L"$ $Learned")
lines!(ax_hq, q_range, H_cont, linewidth=6, color=:black, linestyle=:dash, label=L"$ $Continuum limit")
axislegend(ax_hq, position=:rt)

lines!(ax_eq, q_range, E_sol, linewidth=6, color=:red, linestyle=:solid, label=L"$ $Learned")
lines!(ax_eq, q_range, E_cont, linewidth=6, color=:black, linestyle=:dash, label=L"$ $Continuum limit")
axislegend(ax_eq, position=:rt)
fig
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

