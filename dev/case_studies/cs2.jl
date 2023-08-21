# # Case Studies: Case Study 2 
# 

# This example shows how we obtained the results in the paper 
# for the second case study. Let us load in the package we will need.
using StepwiseEQL
using CairoMakie
using EpithelialDynamics1D
using OrdinaryDiffEq
using Setfield
using LinearSolve
using ReferenceTests #src
fig_path = joinpath(@__DIR__, "figures") #src

# ## Simulating 
# Let us start by simulating the cell dynamics. We use the force law $F(\ell) = k(s - \ell)$ as 
# usual.
k, η, s = 50.0, 1.0, 1 / 5
force_law = (δ, p) -> p.k * (p.s - δ)
force_law_parameters = (k=k, s=s)
initial_condition = LinRange(0, 5, 60) |> collect
prob = CellProblem(;
    force_law,
    force_law_parameters,
    final_time=100.0,
    damping_constant=η,
    initial_condition,
    fix_right=false)
sol = solve(prob, Tsit5(), saveat=LinRange(0, 100.0, 1000))

# ## Equation learning 
# We now define the equation learning problem. The mechanisms to learn are $D(q)$, $H(q)$, and 
# $E(q)$. The basis functions we use are:
diffusion_basis = PolynomialBasis(-1, -3)
rhs_basis = PolynomialBasis(1, 5)
moving_boundary_basis = PolynomialBasis(-1, -3)

# We now learn the equations. Our first attempt is below, where we use `initial=:none` so that we start 
# with no coefficients initially active.
eql_sol = stepwise_selection(sol; diffusion_basis, rhs_basis, moving_boundary_basis,
    mesh_points=500, initial=:none, threshold_tol=(q=0.35,))

# This result is clearly not perfect. To improve this, we limit the time range. We also 
# re-simulate the cell dynamics with a saving frequency. 
sol = solve(prob, Tsit5(), saveat=15 // 199)
eql_sol = stepwise_selection(sol; diffusion_basis, rhs_basis, moving_boundary_basis,
    mesh_points=500, initial=:none, threshold_tol=(q=0.35,),
    time_range=(0.0, 15.0))

# These results have improved slightly. To plot them, we use the following function.
function plot_results(eql_sol, sol, k, s, η, diffusion_basis, rhs_basis, moving_boundary_basis, conserve_mass=false)
    t = (0, 5, 10, 25, 50, 100)
    prob = sol.prob.p
    prob = @set prob.final_time = 100.0
    sol = solve(prob, Tsit5(), saveat=[collect(t); LinRange(0, 100, 2500)] |> sort |> unique)
    time_indices = [findlast(≤(τ), sol.t) for τ in t]
    colors = (:black, :red, :blue, :green, :orange, :purple, :brown)

    pde = eql_sol.pde
    pde = @set pde.final_time = 100.0 # need to resolve so that we plot over the complete time interval
    pde_sol = solve(pde, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_continuum = deepcopy(pde)
    pde_continuum.diffusion_parameters.θ .= [0, k / η, 0]
    pde_continuum.boundary_conditions.rhs.p.θ .= [0, 2.0, -2.0s, 0, 0]
    pde_continuum.boundary_conditions.moving_boundary.p.θ .= [0, k / η, 0]
    pde_ξ = pde_continuum.geometry.mesh_points
    pde_L = pde_sol[end, :]
    pde_q = pde_sol[begin:(end-1), :]
    cell_q = node_densities.(sol.u)
    cell_r = sol.u
    cell_L = sol[end, :]

    q_range = LinRange(5, 12, 250)

    fig = Figure(fontsize=45, resolution=(2220, 961))
    top_grid = fig[1, 1] = GridLayout(1, 2)
    bottom_grid = fig[2, 1] = GridLayout(1, 3)

    ax_pde = Axis(top_grid[1, 1], xlabel=L"x", ylabel=L"q(x,t)", width=950, height=300,
        title=L"(a):$ $ PDE comparison", titlealign=:left,
        xticks=(0:5:15, [L"%$s" for s in 0:5:15]), yticks=(0:5:15, [L"%$s" for s in 0:5:15]))
    for (j, i) in enumerate(time_indices)
        lines!(ax_pde, pde_ξ * pde_L[i], pde_q[:, i], color=colors[j], linestyle=:dash, linewidth=8)
        lines!(ax_pde, cell_r[i], cell_q[i], color=colors[j], linewidth=4, label=L"%$(t[j])")
    end
    arrows!(ax_pde, [6.0], [12.0], [4.0], [-3.0], color=:black, linewidth=8, arrowsize=40)
    text!(ax_pde, [8.0], [11.0], text=L"t", color=:black, fontsize=63)
    xlims!(ax_pde, 0, 15)
    ylims!(ax_pde, 4, 16)

    ax_leading_edge = Axis(top_grid[1, 2], xlabel=L"t", ylabel=L"L(t)", width=950, height=300,
        title=L"(b):$ $ Leading edge", titlealign=:left,
        xticks=(0:10:100, [L"%$s" for s in 0:10:100]), yticks=(0:5:15, [L"%$s" for s in 0:5:15]))
    lines!(ax_leading_edge, pde_sol.t, pde_L, color=:red, linestyle=:dash, linewidth=5, label=L"$ $Learned")
    lines!(ax_leading_edge, sol.t, cell_L, color=:black, linewidth=3, label=L"$ $Discrete")
    axislegend(position=:rb)
    xlims!(ax_leading_edge, 0, 100)
    ylims!(ax_leading_edge, 0, 15)

    ax_diffusion = Axis(bottom_grid[1, 1],
        xlabel=L"q", ylabel=L"D(q)", width=600, height=300,
        title=L"(c): $D(q)$ comparison", titlealign=:left,
        xticks=(5:3:12, [L"%$s" for s in 5:3:12]), yticks=(0:5, [L"%$s" for s in 0:5]))
    D_cont_fnc = q -> (k / η) / q^2
    D_cont = D_cont_fnc.(q_range)
    local D_sol
    try
        D_sol = diffusion_basis.(q_range, Ref(eql_sol.diffusion_theta), Ref(nothing))
    catch e
        print(e)
        D_sol = D_cont
    end
    lines!(ax_diffusion, q_range, D_sol, linewidth=6, color=:red, linestyle=:solid, label=L"$ $Learned")
    lines!(ax_diffusion, q_range, D_cont, linewidth=6, color=:black, linestyle=:dashdot, label=L"$ $Continuum limit")
    axislegend(position=:rt)
    xlims!(ax_diffusion, 5, 12)
    ylims!(ax_diffusion, 0, 5)

    ax_rhs = Axis(bottom_grid[1, 2],
        xlabel=L"q", ylabel=L"H(q)", width=600, height=300,
        title=L"(d): $H(q)$ comparison", titlealign=:left,
        xticks=(5:3:12, [L"%$s" for s in 5:3:12]), yticks=(-100:40:20, [L"%$s" for s in -100:40:20]))
    RHS_cont_fnc = q -> 2q^2 * (1 - s * q)
    RHS_cont = RHS_cont_fnc.(q_range)
    local RHS_sol
    try
        RHS_sol = rhs_basis.(q_range, Ref(eql_sol.rhs_theta), Ref(nothing))
    catch e
        print(e)
        RHS_sol = RHS_cont
    end
    lines!(ax_rhs, q_range, RHS_sol, linewidth=6, color=:red, linestyle=:solid, label=L"$ $Learned")
    lines!(ax_rhs, q_range, RHS_cont, linewidth=6, color=:black, linestyle=:dashdot, label=L"$ $Continuum limit")
    axislegend(position=:rt)
    xlims!(ax_rhs, 5, 12)
    ylims!(ax_rhs, -100, 20)

    ax_moving_boundary = Axis(bottom_grid[1, 3],
        xlabel=L"q", ylabel=L"E(q)", width=600, height=300,
        title=L"(e): $E(q)$ comparison", titlealign=:left,
        xticks=(5:3:12, [L"%$s" for s in 5:3:12]), yticks=(0:5, [L"%$s" for s in 0:5]))
    MB_cont_fnc = q -> (k / η) / q^2
    MB_cont = MB_cont_fnc.(q_range)
    local MB_sol
    try
        if !conserve_mass
            MB_sol = moving_boundary_basis.(q_range, Ref(eql_sol.moving_boundary_theta), Ref(nothing))
        else
            MB_sol = moving_boundary_basis.(q_range, Ref(eql_sol.diffusion_theta), Ref(nothing))
        end
    catch e
        print(e)
        MB_sol = MB_cont
    end
    lines!(ax_moving_boundary, q_range, MB_sol, linewidth=6, color=:red, linestyle=:solid, label=L"$ $Learned")
    lines!(ax_moving_boundary, q_range, MB_cont, linewidth=6, color=:black, linestyle=:dashdot, label=L"$ $Continuum limit")
    axislegend(position=:rt)
    xlims!(ax_moving_boundary, 5, 12)
    ylims!(ax_moving_boundary, 0, 5)
    fig
end
nothing #hide

# Using this function, we obtain the plot below.
fig = plot_results(eql_sol, sol, k, s, η, diffusion_basis, rhs_basis, moving_boundary_basis)
save(joinpath(fig_path, "figure6.pdf"), fig) #src
@test_reference joinpath(fig_path, "figure6.png") fig #src

# Since the leading edges start to diverge for late time, we need to prune the matrix slightly. 
# We do this as follows, giving our improved results.
eql_sol = stepwise_selection(sol; diffusion_basis, rhs_basis, moving_boundary_basis,
    mesh_points=500, initial=:none, threshold_tol=(q=0.35, dL=0.1),
    time_range=(0.0, 15.0))

#-
fig = plot_results(eql_sol, sol, k, s, η, diffusion_basis, rhs_basis, moving_boundary_basis)
save(joinpath(fig_path, "figure7.pdf"), fig) #src
@test_reference joinpath(fig_path, "figure7.png") fig #src

# ## Conservation of mass 
# If we want to enforce conservation of mass, we can set $D(q) = E(q)$ by simply using 
# the keyword argument `conserve_mass=true` as below. These are the results that we 
# describe in more detail in the supplementary material of our paper.
eql_sol = stepwise_selection(sol; diffusion_basis, rhs_basis, moving_boundary_basis,
    mesh_points=500, initial=:none, threshold_tol=(q=0.35, dL=0.1),
    time_range=(0.0, 15.0), conserve_mass=true)

#-
fig = plot_results(eql_sol, sol, k, s, η, diffusion_basis, rhs_basis, moving_boundary_basis, true)
save(joinpath(fig_path, "sfigure_conserve.pdf"), fig) #src
@test_reference joinpath(fig_path, "sfigure_conserve.png") fig #src

