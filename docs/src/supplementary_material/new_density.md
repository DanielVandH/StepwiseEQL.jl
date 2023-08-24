```@meta
EditURL = "https://github.com/DanielVandH/StepwiseEQL.jl/tree/main/docs/src/supplementary_material/new_density.md"
```

!!! tip
    This example is  also available as a Jupyter notebook:
    [`new_density.ipynb`](https://github.com/DanielVandH/StepwiseEQL.jl/tree/main/docs/src/supplementary_material/notebooks/new_density.ipynb)

# Discrete Densities at the Boundaries

In this section, we show how we obtained the figures in the paper that
motivate the need for our new definition of density at the boundary. We recall
that we defined
```math
\begin{align*}
q_1(t) &= \frac{2}{x_2(t) - x_1(t)} - \frac{2}{x_3(t) - x_1(t)}, \\
q_n(t) &= \frac{2}{x_n(t) - x_{n-1}(t)} - \frac{2}{x_n(t) - x_{n-2}(t)},
\end{align*}
```
modifying the previous definitions $q_1(t) = 1/x_2(t)$ and $q_n(t) = 1/(x_n(t) - x_{n-1}(t))$.

To start, let us load in the packages we will need.

```julia
using StepwiseEQL
using CairoMakie
using EpithelialDynamics1D
using OrdinaryDiffEq
using LinearSolve
```

We now define and solve the `CellProblem`, and its
corresponding continuum limit.

```julia
force_law = (δ, p) -> p.k * (p.s - δ)
k, s, η, T = 50.0, 0.2, 1.0, 10.0
force_law_parameters = (k=k, s=s)
initial_condition = collect(LinRange(0, 5, 30))
prob = CellProblem(;
    force_law,
    force_law_parameters,
    final_time=T,
    damping_constant=η,
    initial_condition,
    fix_right=false)
sol = solve(prob, Tsit5(), saveat=0.02)
pde_prob = continuum_limit(prob, 5000, proliferation=false)
pde_sol = solve(pde_prob, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
```
```
retcode: Success
Interpolation: 1st order linear
t: 501-element Vector{Float64}:
  0.0
  0.02
  0.04
  0.06
  0.08
  ⋮
  9.94
  9.96
  9.98
 10.0
u: 501-element Vector{Vector{Float64}}:
 [5.8, 5.8, 5.8, 5.8, 5.8, 5.8, 5.799999999999999, 5.800000000000001, 5.8, 5.800000000000001  …  5.799999999999996, 5.799999999999996, 5.799999999999996, 5.799999999999996, 5.799999999999996, 5.799999999999996, 5.799999999999996, 5.799999999999996, 5.799999999999996, 5.0]
 [5.799999999999999, 5.799999999999999, 5.800000000000001, 5.800000000000001, 5.799999999999999, 5.8, 5.8, 5.800000000000001, 5.799999999999999, 5.799999999999999  …  5.1924095336557405, 5.190495704337308, 5.188581949234203, 5.186668313326301, 5.184756020440565, 5.182847493803263, 5.1809447979123915, 5.17904652757611, 5.177146783848216, 5.020959142248633]
 [5.799999999999999, 5.8, 5.799999999999999, 5.799999999999999, 5.8, 5.8, 5.8, 5.800000000000001, 5.8, 5.799999999999999  …  5.143442394138647, 5.142046298632507, 5.140650182859538, 5.139254077827552, 5.137858288677557, 5.136463403388777, 5.1350699293367486, 5.133677552337702, 5.132284878470131, 5.032916286859433]
 [5.799999999999995, 5.799999999999995, 5.799999999999995, 5.7999999999999945, 5.799999999999995, 5.799999999999995, 5.799999999999995, 5.799999999999995, 5.799999999999995, 5.7999999999999945  …  5.118227792139378, 5.117079988404248, 5.115932332278834, 5.114784836161163, 5.113637590358396, 5.112490765195386, 5.111344507534536, 5.110198730756248, 5.109053042509071, 5.042421082805959]
 [5.79999999999663, 5.799999999996631, 5.799999999996631, 5.79999999999663, 5.79999999999663, 5.799999999996629, 5.799999999996628, 5.799999999996627, 5.799999999996626, 5.7999999999966265  …  5.103163998482414, 5.102163055749991, 5.1011623632269725, 5.10016193038084, 5.099161837944899, 5.098162240724732, 5.097163272645574, 5.0961648508727775, 5.095166605712397, 5.050446188390052]
 ⋮
 [5.2225071190676555, 5.222507107636834, 5.22250707334437, 5.22250701619027, 5.222506936174542, 5.222506833297196, 5.222506707558247, 5.222506558957714, 5.222506387495615, 5.222506193171979  …  5.006205275518905, 5.00614107114595, 5.006076868698485, 5.006012668182211, 5.0059484696028385, 5.005884272966066, 5.005820078277598, 5.005755885543136, 5.005691694768377, 5.639095864107762]
 [5.22184873206438, 5.221848720669727, 5.2218486864857665, 5.221848629512506, 5.221848549749951, 5.221848447198115, 5.221848321857011, 5.221848173726659, 5.221848002807076, 5.221847809098291  …  5.006188207936062, 5.006124180483525, 5.006060154940927, 5.0059961313139505, 5.005932109608278, 5.005868089829588, 5.005804071983562, 5.005740056075877, 5.00567604211221, 5.639550474565784]
 [5.221192591559284, 5.221192580200655, 5.22119254612477, 5.221192489331633, 5.2211924098212545, 5.221192307593643, 5.2211921826488155, 5.221192034986788, 5.22119186460758, 5.221191671511219  …  5.0061711520945495, 5.006107300909339, 5.006043451618299, 5.005979604227087, 5.005915758741366, 5.005851915166788, 5.005788073509013, 5.005724233773695, 5.005660395966485, 5.64000372631041]
 [5.2205386932976285, 5.220538681974883, 5.220538648006646, 5.220538591392923, 5.220538512133723, 5.220538410229056, 5.220538285678937, 5.220538138483384, 5.220537968642416, 5.22053777615606  …  5.006154106765068, 5.006090431192137, 5.0060267574973825, 5.005963085686441, 5.005899415764949, 5.005835747738538, 5.00577208161284, 5.005708417393488, 5.005644755086109, 5.640455621425953]
```

Next, we need to get the data for the densities at $t=2$,
as well as the derivatives. The densities are obtained below.

```julia
t_idx = findlast(≤(2), sol.t)
new_densities = node_densities(sol.u[t_idx])
baker_densities = copy(new_densities)
baker_densities[begin] = 1 / (sol.u[t_idx][2] - sol.u[t_idx][1])
baker_densities[end] = 1 / (sol.u[t_idx][end] - sol.u[t_idx][end-1])
```
```
5.041919353189308
```

To now set up the derivatives, we need the $(x, t)$ data.

```julia
pde_L = pde_sol.u[t_idx][end]
pde_q = pde_sol.u[t_idx][begin:(end-1)]
pde_x = pde_prob.geometry.mesh_points * pde_L
```
```
5000-element Vector{Float64}:
 0.0
 0.0010598254070783829
 0.0021196508141567657
 0.0031794762212351488
 0.004239301628313531
 ⋮
 5.294887733763601
 5.2959475591706795
 5.297007384577758
 5.298067209984836
```

Next, we compute all the data.

```julia
new_dx = zeros(length(sol))
baker_dx = zeros(length(sol))
continuum_dx = zeros(length(sol))
for j in eachindex(sol)
    new_dx[j] = StepwiseEQL.cell_∂q∂x(sol, length(sol.u[j]), j)
    qₙ₋₂ = StepwiseEQL.cell_density(sol, length(sol.u[j]) - 2, j)
    qₙ₋₁ = StepwiseEQL.cell_density(sol, length(sol.u[j]) - 1, j)
    qₙ = 1 / (sol.u[j][end] - sol.u[j][end-1])
    xₙ₋₂ = sol.u[j][end-2]
    xₙ₋₁ = sol.u[j][end-1]
    xₙ = sol.u[j][end]
    baker_dx[j] = StepwiseEQL.backward_dfdx(qₙ₋₂, qₙ₋₁, qₙ, xₙ₋₂, xₙ₋₁, xₙ)
    baker_dx[j] = (qₙ - qₙ₋₁) / (xₙ - xₙ₋₁)
    u = pde_sol.u[j][end-1]
    continuum_dx[j] = 2u^2 * (1 - u * s)
end
```

Finally, we can plot the comparisons.

```julia
fig = Figure(fontsize=43, resolution=(1580, 950))
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"q(x,t)",
    width=600, height=300,
    xticks=(0:6, [L"%$s" for s in 0:6]),
    yticks=(5:0.2:5.8, [L"%$s" for s in 5:0.2:5.8]),
    title=L"(a):$ $ Complete view",
    titlealign=:left)
lines!(ax, sol.u[t_idx], new_densities, linewidth=3, color=:red, label=L"$ $New")
lines!(ax, sol.u[t_idx], baker_densities, linewidth=3, color=:blue, label=L"$ $Baker")
lines!(ax, pde_x, pde_q, linewidth=3, color=:black, label=L"$ $Continuum limit")
lines!(ax, [(5.0, 5.0), (5.35, 5.0), (5.35, 5.1), (5.0, 5.1), (5.0, 5.0)], color = :magenta, linewidth=4)
ax = Axis(fig[1, 2], xlabel=L"x", ylabel=L"q(x,t)",
    width=600, height=300,
    title=L"(b):$ $ Focused view",
    xticks=(5:0.1:5.3, [L"%$s" for s in 5:0.1:5.3]),
    yticks=(5:0.05:5.1, [L"%$s" for s in 5:0.05:5.1]),
    titlealign=:left)
lines!(ax, sol.u[t_idx], new_densities, linewidth=3, color=:red, label=L"$ $New")
lines!(ax, sol.u[t_idx], baker_densities, linewidth=3, color=:blue, label=L"$ $Baker")
lines!(ax, pde_x, pde_q, linewidth=3, color=:black, label=L"$ $Continuum limit")
xlims!(ax, 5, 5.3)
ylims!(ax, 5, 5.1)
ax = Axis(fig[2, 1:2],
    xlabel=L"t", ylabel=L"\partial q/\partial x",
    title=L"(c):$ $ Derivative comparisons",
    titlealign=:left,
    xticks=(0:2:10, [L"%$s" for s in 0:2:10]),
    yticks=(-1.5:0.5:0, [L"%$s" for s in -1.5:0.5:0]),
    width=1350,
    height=300)
lines!(ax, sol.t[2:end], new_dx[2:end], linewidth=3, color=:red, label=L"$ $New")
lines!(ax, sol.t[2:end], baker_dx[2:end], linewidth=3, color=:blue, label=L"$ $Baker")
lines!(ax, sol.t[2:end], continuum_dx[2:end], linewidth=3, color=:black, label=L"$ $Continuum limit")
xlims!(ax, 0, 10)
ylims!(ax, -1.5, 0.2)
axislegend(ax, position=:rb)
fig
```
```@raw html
<figure>
    <img src='../../figures/sfigure_boundary_density.png', alt='Figure S1 from the paper'><br>
</figure>
```

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/DanielVandH/StepwiseEQL.jl/tree/main/docs/src/literate_supplementary_material/new_density.jl).

```julia
using StepwiseEQL
using CairoMakie
using EpithelialDynamics1D
using OrdinaryDiffEq
using LinearSolve

force_law = (δ, p) -> p.k * (p.s - δ)
k, s, η, T = 50.0, 0.2, 1.0, 10.0
force_law_parameters = (k=k, s=s)
initial_condition = collect(LinRange(0, 5, 30))
prob = CellProblem(;
    force_law,
    force_law_parameters,
    final_time=T,
    damping_constant=η,
    initial_condition,
    fix_right=false)
sol = solve(prob, Tsit5(), saveat=0.02)
pde_prob = continuum_limit(prob, 5000, proliferation=false)
pde_sol = solve(pde_prob, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)

t_idx = findlast(≤(2), sol.t)
new_densities = node_densities(sol.u[t_idx])
baker_densities = copy(new_densities)
baker_densities[begin] = 1 / (sol.u[t_idx][2] - sol.u[t_idx][1])
baker_densities[end] = 1 / (sol.u[t_idx][end] - sol.u[t_idx][end-1])

pde_L = pde_sol.u[t_idx][end]
pde_q = pde_sol.u[t_idx][begin:(end-1)]
pde_x = pde_prob.geometry.mesh_points * pde_L

new_dx = zeros(length(sol))
baker_dx = zeros(length(sol))
continuum_dx = zeros(length(sol))
for j in eachindex(sol)
    new_dx[j] = StepwiseEQL.cell_∂q∂x(sol, length(sol.u[j]), j)
    qₙ₋₂ = StepwiseEQL.cell_density(sol, length(sol.u[j]) - 2, j)
    qₙ₋₁ = StepwiseEQL.cell_density(sol, length(sol.u[j]) - 1, j)
    qₙ = 1 / (sol.u[j][end] - sol.u[j][end-1])
    xₙ₋₂ = sol.u[j][end-2]
    xₙ₋₁ = sol.u[j][end-1]
    xₙ = sol.u[j][end]
    baker_dx[j] = StepwiseEQL.backward_dfdx(qₙ₋₂, qₙ₋₁, qₙ, xₙ₋₂, xₙ₋₁, xₙ)
    baker_dx[j] = (qₙ - qₙ₋₁) / (xₙ - xₙ₋₁)
    u = pde_sol.u[j][end-1]
    continuum_dx[j] = 2u^2 * (1 - u * s)
end

fig = Figure(fontsize=43, resolution=(1580, 950))
ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"q(x,t)",
    width=600, height=300,
    xticks=(0:6, [L"%$s" for s in 0:6]),
    yticks=(5:0.2:5.8, [L"%$s" for s in 5:0.2:5.8]),
    title=L"(a):$ $ Complete view",
    titlealign=:left)
lines!(ax, sol.u[t_idx], new_densities, linewidth=3, color=:red, label=L"$ $New")
lines!(ax, sol.u[t_idx], baker_densities, linewidth=3, color=:blue, label=L"$ $Baker")
lines!(ax, pde_x, pde_q, linewidth=3, color=:black, label=L"$ $Continuum limit")
ax = Axis(fig[1, 2], xlabel=L"x", ylabel=L"q(x,t)",
    width=600, height=300,
    title=L"(b):$ $ Focused view",
    xticks=(5:0.1:5.3, [L"%$s" for s in 5:0.1:5.3]),
    yticks=(5:0.05:5.1, [L"%$s" for s in 5:0.05:5.1]),
    titlealign=:left)
lines!(ax, sol.u[t_idx], new_densities, linewidth=3, color=:red, label=L"$ $New")
lines!(ax, sol.u[t_idx], baker_densities, linewidth=3, color=:blue, label=L"$ $Baker")
lines!(ax, pde_x, pde_q, linewidth=3, color=:black, label=L"$ $Continuum limit")
xlims!(ax, 5, 5.3)
ylims!(ax, 5, 5.1)
ax = Axis(fig[2, 1:2],
    xlabel=L"t", ylabel=L"\partial q/\partial x",
    title=L"(c):$ $ Derivative comparisons",
    titlealign=:left,
    xticks=(0:2:10, [L"%$s" for s in 0:2:10]),
    yticks=(-1.5:0.5:0, [L"%$s" for s in -1.5:0.5:0]),
    width=1350,
    height=300)
lines!(ax, sol.t[2:end], new_dx[2:end], linewidth=3, color=:red, label=L"$ $New")
lines!(ax, sol.t[2:end], baker_dx[2:end], linewidth=3, color=:blue, label=L"$ $Baker")
lines!(ax, sol.t[2:end], continuum_dx[2:end], linewidth=3, color=:black, label=L"$ $Continuum limit")
xlims!(ax, 0, 10)
ylims!(ax, -1.5, 0.2)
axislegend(ax, position=:rb)
fig
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

