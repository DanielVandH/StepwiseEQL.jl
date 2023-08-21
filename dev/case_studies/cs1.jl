# # Case Studies: Case Study 1
# 

# This example shows how we obtained the results in the paper for the first case study. To start, 
# let us load in the packages we will need.
using StepwiseEQL
using CairoMakie
using EpithelialDynamics1D
using OrdinaryDiffEq
using ReferenceTests #src
fig_path = joinpath(@__DIR__, "figures") #src

# ## Simulating 
# The first step in our procedure is to obtain the results from the cell simulation. This is 
# done as follows. We use the force law $F(\ell) = k(s - \ell)$.
force_law = (δ, p) -> p.k * (p.s - δ)
force_law_parameters = (k=50.0, s=0.2)
final_time = 5.0
damping_constant = 1.0
initial_condition = [LinRange(0, 5, 30); LinRange(25, 30, 30)] |> unique!
prob = CellProblem(;
    force_law,
    force_law_parameters,
    final_time,
    damping_constant,
    initial_condition)
sol = solve(prob, Tsit5(), saveat=LinRange(0, final_time, 50))

# ## Equation learning
# To now define the equation learning problem, we note that all we need to learn 
# is $D(q)$. The basis expansion we use is 
# ```math
# D(q) = \dfrac{\theta_1^d}{q} + \dfrac{\theta_2^d}{q^2} + \dfrac{\theta_3^d}{q^3},
# ```
# which we can define as follows:
diffusion_basis = BasisSet(
    (q, k) -> inv(q),
    (q, k) -> inv(q^2),
    (q, k) -> inv(q^3),
)

# This could have also been defined using 
diffusion_basis = PolynomialBasis(-1, -3)

# which is simpler. Next, let us obtain the results. The call to `stepwise_selection` 
# is simple in this case. To start, we use no pruning:
eql_sol = stepwise_selection(sol; diffusion_basis)

# The coefficient for $\theta_2^d$ is not perfect. If we instead use some pruning on $q$, 
# we can obtain an improved result:
eql_sol2 = stepwise_selection(sol; diffusion_basis, threshold_tol=(q=0.1,))

# (Note that the comma after `0.1` is necessary so that we get a `NamedTuple`, otherwise it doesn't parse as a `Tuple`. Compare:
threshold_tol = (q = 0.1) # same as threshold_tol = q = 0.1

#-
threshold_tol = (q=0.1,)

# This is only for `NamedTuple`s with a single element, e.g. `(a = 0.1, b = 0.2)` is fine.) We note also that if you 
# want the LaTeX form of these tables, for `eql_sol` you could use for example:
latex_table(eql_sol)

# ## Plotting 
# ### Progression of the Diffusion Curves 
# Let us now examine our results. First, we see how the diffusion curve is changed at each step of our procedure, 
# comparing the results with pruning and without pruning.
fig = Figure(fontsize=38)
ax1 = Axis(fig[1, 1], xlabel=L"q", ylabel=L"D(q)",
    width=600, height=300,
    title=L"(a): $D(q)$ progression without pruning", titlealign=:left,
    xticks=(0:2:6, [L"%$s" for s in 0:2:6]),
    yticks=(-20:20:80, [L"%$s" for s in -20:20:80]))
ax2 = Axis(fig[1, 2], xlabel=L"q", ylabel=L"D(q)",
    width=600, height=300,
    title=L"(b): $D(q)$ progression with pruning", titlealign=:left,
    xticks=(0:2:6, [L"%$s" for s in 0:2:6]),
    yticks=(-20:20:80, [L"%$s" for s in -20:20:80]))
q_range = LinRange(1 / 10, 6, 250)
Dθ_no_prune = eql_sol.diffusion_theta_history
Dθ_prune = eql_sol2.diffusion_theta_history
colors = (:red, :black, :lightgreen)
linestyles = (:solid, :solid, :dash)
for j in 1:3
    lines!(ax1, q_range, diffusion_basis.(q_range, Ref(Dθ_no_prune[:, j]), Ref(nothing)), linewidth=6, linestyle=linestyles[j], color=colors[j], label=L"Step $%$(j)$")
    lines!(ax2, q_range, diffusion_basis.(q_range, Ref(Dθ_prune[:, j]), Ref(nothing)), linewidth=6, linestyle=linestyles[j], color=colors[j], label=L"Step $%$(j)$")
end
for ax in (ax1, ax2)
    ylims!(ax, -20, 80)
    hlines!(ax, [0.0], linecolor=:grey, linewidth=3, linestyle=:dash)
end
fig[1, 3] = Legend(fig, ax1)
resize_to_layout!(fig)
fig
save(joinpath(fig_path, "figure4.pdf"), fig) #src
@test_reference joinpath(fig_path, "figure4.png") fig #src

# ### Comparing Density Results
# Let us now compare the density results.
fig = Figure(fontsize=45, resolution=(1470, 961))
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"q(x, t)",
    width=600, height=300,
    title=L"(a):$ $ No pruning", titlealign=:left,
    xticks=(0:10:30, [L"%$s" for s in 0:10:30]),
    yticks=(0:2:6, [L"%$s" for s in 0:2:6]))
ax2 = Axis(fig[1, 2], xlabel=L"x", ylabel=L"q(x, t)",
    width=600, height=300,
    title=L"(b):$ $ Pruning", titlealign=:left,
    xticks=(0:10:30, [L"%$s" for s in 0:10:30]),
    yticks=(0:2:6, [L"%$s" for s in 0:2:6]))
t = (0, 1, 2, 3, 4, 5)
colors = (:black, :red, :blue, :green, :orange, :purple)
time_indices = [findlast(≤(τ), sol.t) for τ in t]
for (j, i) in enumerate(time_indices)
    lines!(ax, eql_sol.pde.geometry.mesh_points, eql_sol.pde_sol.u[i], color=colors[j], linestyle=:dash, linewidth=5)
    lines!(ax, sol.u[i], node_densities(sol.u[i]), color=colors[j], linewidth=3)
    lines!(ax2, eql_sol2.pde.geometry.mesh_points, eql_sol2.pde_sol.u[i], color=colors[j], linestyle=:dash, linewidth=5)
    lines!(ax2, sol.u[i], node_densities(sol.u[i]), color=colors[j], linewidth=3, label=L"%$(t[j])")
end
for ax in (ax, ax2)
    arrows!(ax, [15.0, 23.0], [0.4, 3.0], [0.0, 4.0], [2.0, -2.0], color=:black, linewidth=8, arrowsize=40)
    text!(ax, [15.7, 28.0], [2.0, 0.7], text=[L"t", L"t"], color=:black, fontsize=47)
    xlims!(ax, 0, 30)
end
ax3 = Axis(fig[2, 1:2], xlabel=L"q", ylabel=L"D(q)",
    width=1200, height=300,
    title=L"(c):$ $ Nonlinear diffusivity", titlealign=:left,
    xticks=(0:2:6, [L"%$s" for s in 0:2:6]),
    yticks=(0:20:80, [L"%$s" for s in 0:20:80]))
D_no_prune = diffusion_basis.(q_range, Ref(eql_sol.diffusion_theta), Ref(nothing))
D_prune = diffusion_basis.(q_range, Ref(eql_sol2.diffusion_theta), Ref(nothing))
D_cont_fnc = q -> (force_law_parameters.k / damping_constant) / q^2
D_cont = D_cont_fnc.(q_range)
lines!(ax3, q_range, D_no_prune, linewidth=6, color=:red, linestyle=:solid, label=L"$ $Non-pruned")
lines!(ax3, q_range, D_prune, linewidth=6, color=:black, linestyle=:solid, label=L"$ $Pruned")
lines!(ax3, q_range, D_cont, linewidth=6, color=:lightgreen, linestyle=:dash, label=L"$ $Continuum limit")
axislegend(position=:rt)
ylims!(ax3, 0, 80)
fig
save(joinpath(fig_path, "figure5.pdf"), fig) #src
@test_reference joinpath(fig_path, "figure5.png") fig #src