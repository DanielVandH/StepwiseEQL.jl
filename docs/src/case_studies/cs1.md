```@meta
EditURL = "https://github.com/DanielVandH/StepwiseEQL.jl/tree/main/docs/src/case_studies/cs1.md"
```

!!! tip
    This example is also available as a Jupyter notebook:
    [`cs1.ipynb`](https://github.com/DanielVandH/StepwiseEQL.jl/tree/main/docs/src/case_studies/notebooks/cs1.ipynb)

# Case Studies: Case Study 1

This example shows how we obtained the results in the paper for the first case study. To start,
let us load in the packages we will need.

```julia
using StepwiseEQL
using CairoMakie
using EpithelialDynamics1D
using OrdinaryDiffEq
```

## Simulating
The first step in our procedure is to obtain the results from the cell simulation. This is
done as follows. We use the force law $F(\ell) = k(s - \ell)$.

```julia
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
```
```
retcode: Success
Interpolation: 1st order linear
t: 50-element Vector{Float64}:
 0.0
 0.1020408163265306
 0.2040816326530612
 0.30612244897959184
 0.4081632653061224
 ⋮
 4.6938775510204085
 4.795918367346939
 4.8979591836734695
 5.0
u: 50-element Vector{Vector{Float64}}:
 [0.0, 0.1724137931034483, 0.3448275862068966, 0.5172413793103449, 0.6896551724137931, 0.8620689655172414, 1.0344827586206897, 1.206896551724138, 1.3793103448275863, 1.5517241379310345  …  28.448275862068964, 28.620689655172413, 28.79310344827586, 28.965517241379313, 29.137931034482758, 29.310344827586206, 29.482758620689655, 29.655172413793103, 29.827586206896555, 30.0]
 [0.0, 0.17241379310349564, 0.3448275862072093, 0.5172413793123272, 0.6896551724254885, 0.862068965583525, 1.0344827589757786, 1.2068965535463891, 1.379310353708087, 1.5517241793563723  …  28.448275820643673, 28.620689646291883, 28.793103446453646, 28.96551724102418, 29.137931034416518, 29.31034482757447, 29.4827586206877, 29.655172413792776, 29.827586206896513, 30.0]
 [0.0, 0.1724138117467249, 0.3448276572464026, 0.5172415843189141, 0.6896558354604316, 0.8620707934381211, 1.0344882309935757, 1.2069109680032413, 1.379350391424813, 1.5518243068723196  …  28.44817569312707, 28.62064960857575, 28.793089031996246, 28.965511769006884, 29.137929206561484, 29.31034416453989, 29.48275841568083, 29.655172342753765, 29.827586188253186, 30.0]
 [0.0, 0.1724180723087107, 0.3448393892353834, 0.5172691410859218, 0.6897179116629475, 0.8622065671105769, 1.0347778052706758, 1.2075127763561453, 1.3805674430039012, 1.554222253567815  …  28.445777746432174, 28.619432556996106, 28.792487223643853, 28.965222194729332, 29.137793432889413, 29.310282088337058, 29.482730858914078, 29.65516061076462, 29.827581927691288, 30.0]
 [0.0, 0.17250037129897575, 0.34502183801768355, 0.5176649951705653, 0.6904162692664148, 0.8635599798065833, 1.0370852193558056, 1.2116982876663236, 1.3874720675876706, 1.5660206102285164  …  28.4339793897718, 28.612527932412014, 28.788301712333965, 28.96291478064394, 29.13644002019364, 29.309583730733404, 29.482335004829572, 29.65497816198222, 29.827499628701077, 30.0]
 ⋮
 [0.0, 0.4614393800663206, 0.9236739347032462, 1.386439368972283, 1.851563420601704, 2.3177075992535756, 2.7876942156439184, 3.2591089912813254, 3.735720931301201, 4.214052274597669  …  25.78594772540233, 26.264279068698805, 26.740891008718663, 27.212305784356094, 27.682292400746412, 28.148436579398304, 28.613560631027713, 29.07632606529675, 29.538560619933687, 30.0]
 [0.0, 0.46426752760006446, 0.9285313930002487, 1.3948029724410733, 1.86104668264444, 2.331249447002487, 2.8013504305271613, 3.2772511919669443, 3.7528965222099173, 4.236021671466421  …  25.763978328533593, 26.247103477790063, 26.72274880803307, 27.19864956947283, 27.66875055299752, 28.13895331735556, 28.60519702755893, 29.07146860699975, 29.535732472399932, 30.0]
 [0.0, 0.4667997920222022, 0.933391698910398, 1.4022871779794028, 1.8705453674745856, 2.3433528371963184, 2.815053577475178, 3.2934356472551616, 3.7701776720962408, 4.255567677743931  …  25.744432322256078, 26.229822327903747, 26.70656435274485, 27.184946422524817, 27.656647162803683, 28.129454632525412, 28.5977128220206, 29.066608301089605, 29.5332002079778, 30.0]
 [0.0, 0.46888429127863634, 0.938590802821992, 1.4084378023194548, 1.8807270563962621, 2.3532656659043787, 2.8297942955536013, 3.306619606648261, 3.7888653821560094, 4.271367024660429  …  25.728632975339583, 26.211134617843985, 26.69338039335174, 27.170205704446403, 27.64673433409563, 28.119272943603733, 28.591562197680545, 29.061409197178005, 29.53111570872137, 30.0]
```

## Equation learning
To now define the equation learning problem, we note that all we need to learn
is $D(q)$. The basis expansion we use is
```math
D(q) = \dfrac{\theta_1^d}{q} + \dfrac{\theta_2^d}{q^2} + \dfrac{\theta_3^d}{q^3},
```
which we can define as follows:

```julia
diffusion_basis = BasisSet(
    (q, k) -> inv(q),
    (q, k) -> inv(q^2),
    (q, k) -> inv(q^3),
)
```
```
(::BasisSet{Tuple{Main.var"##19533".var"#3#6", Main.var"##19533".var"#4#7", Main.var"##19533".var"#5#8"}}) (generic function with 3 methods)
```

This could have also been defined using

```julia
diffusion_basis = PolynomialBasis(-1, -3)
```
```
(::BasisSet{Tuple{StepwiseEQL.var"#52#54"{Int64}, StepwiseEQL.var"#52#54"{Int64}, StepwiseEQL.var"#52#54"{Int64}}}) (generic function with 3 methods)
```

which is simpler. Next, let us obtain the results. The call to `stepwise_selection`
is simple in this case. To start, we use no pruning:

```julia
eql_sol = stepwise_selection(sol; diffusion_basis)
```
```
StepwiseEQL Solution.
    D(q) = θ₂ᵈ ϕ₂ᵈ(q)
┌──────┬──────────────────────┬───────┐
│ Step │   θ₁ᵈ    θ₂ᵈ     θ₃ᵈ │  Loss │
├──────┼──────────────────────┼───────┤
│    1 │ -5.97  70.73  -27.06 │   Inf │
│    2 │ -1.46  47.11    0.00 │ -4.33 │
│    3 │  0.00  43.52    0.00 │ -5.18 │
└──────┴──────────────────────┴───────┘
```

The coefficient for $\theta_2^d$ is not perfect. If we instead use some pruning on $q$,
we can obtain an improved result:

```julia
eql_sol2 = stepwise_selection(sol; diffusion_basis, threshold_tol=(q=0.1,))
```
```
StepwiseEQL Solution.
    D(q) = θ₂ᵈ ϕ₂ᵈ(q)
┌──────┬─────────────────────┬───────┐
│ Step │   θ₁ᵈ    θ₂ᵈ    θ₃ᵈ │  Loss │
├──────┼─────────────────────┼───────┤
│    1 │ -1.45  42.48  13.76 │ -4.19 │
│    2 │  0.00  37.79  19.69 │ -5.46 │
│    3 │  0.00  49.83   0.00 │ -7.97 │
└──────┴─────────────────────┴───────┘
```

(Note that the comma after `0.1` is necessary so that we get a `NamedTuple`, otherwise it doesn't parse as a `Tuple`. Compare:

```julia
threshold_tol = (q = 0.1) # same as threshold_tol = q = 0.1
```
```
0.1
```

```julia
threshold_tol = (q=0.1,)
```
```
(q = 0.1,)
```

This is only for `NamedTuple`s with a single element, e.g. `(a = 0.1, b = 0.2)` is fine.) We note also that if you
want the LaTeX form of these tables, for `eql_sol` you could use for example:

```julia
latex_table(eql_sol)
```
```
StepwiseEQL Solution.
    D(q) = θ₂ᵈ ϕ₂ᵈ(q)
\begin{tabular}{|r|rrr|r|}
  \hline
  \textbf{Step} & \textbf{$\theta_{1}^d$ } & \textbf{$\theta_{2}^d$ } & \textbf{$\theta_{3}^d$ } & \textbf{Loss} \\\hline
  1 & -5.97 & 70.73 & \color{blue}{\textbf{-27.06}} & $\infty$ \\
  2 & \color{blue}{\textbf{-1.46}} & 47.11 & 0.00 & -4.33 \\
  3 & 0.00 & 43.52 & 0.00 & -5.18 \\\hline
\end{tabular}
```

## Plotting
### Progression of the Diffusion Curves
Let us now examine our results. First, we see how the diffusion curve is changed at each step of our procedure,
comparing the results with pruning and without pruning.

```julia
fig = Figure(fontsize=38)
ax1 = Axis(fig[1, 1], xlabel=L"q", ylabel=L"D(q)",
    width=600, height=300,
    title=L"(a): $D(q)$ without pruning", titlealign=:left,
    xticks=(0:2:6, [L"%$s" for s in 0:2:6]),
    yticks=(-20:20:80, [L"%$s" for s in -20:20:80]))
ax2 = Axis(fig[1, 2], xlabel=L"q", ylabel=L"D(q)",
    width=600, height=300,
    title=L"(b): $D(q)$ with pruning", titlealign=:left,
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
    hlines!(ax, [0.0], color=:grey, linewidth=6, linestyle=:dash)
end
fig[1, 3] = Legend(fig, ax1)
resize_to_layout!(fig)
fig
```
```@raw html
<figure>
    <img src='../../figures/figure4.png', alt='Figure 4 from the paper'><br>
</figure>
```

### Comparing Density Results
Let us now compare the density results.

```julia
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
    title=L"(c): $D(q)$ comparison", titlealign=:left,
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
```

```@raw html
<figure>
    <img src='../../figures/figure5.png', alt='Figure 5 from the paper'><br>
</figure>
```

## Just the code
An uncommented version of this example is given below.
You can view the source code for this file [here](https://github.com/DanielVandH/StepwiseEQL.jl/tree/main/docs/src/literate_case_studies/cs1.jl).

```julia
using StepwiseEQL
using CairoMakie
using EpithelialDynamics1D
using OrdinaryDiffEq

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

diffusion_basis = BasisSet(
    (q, k) -> inv(q),
    (q, k) -> inv(q^2),
    (q, k) -> inv(q^3),
)

diffusion_basis = PolynomialBasis(-1, -3)

eql_sol = stepwise_selection(sol; diffusion_basis)

eql_sol2 = stepwise_selection(sol; diffusion_basis, threshold_tol=(q=0.1,))

threshold_tol = (q = 0.1) # same as threshold_tol = q = 0.1

threshold_tol = (q=0.1,)

latex_table(eql_sol)

fig = Figure(fontsize=38)
ax1 = Axis(fig[1, 1], xlabel=L"q", ylabel=L"D(q)",
    width=600, height=300,
    title=L"(a): $D(q)$ without pruning", titlealign=:left,
    xticks=(0:2:6, [L"%$s" for s in 0:2:6]),
    yticks=(-20:20:80, [L"%$s" for s in -20:20:80]))
ax2 = Axis(fig[1, 2], xlabel=L"q", ylabel=L"D(q)",
    width=600, height=300,
    title=L"(b): $D(q)$ with pruning", titlealign=:left,
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
    hlines!(ax, [0.0], color=:grey, linewidth=6, linestyle=:dash)
end
fig[1, 3] = Legend(fig, ax1)
resize_to_layout!(fig)
fig

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
    title=L"(c): $D(q)$ comparison", titlealign=:left,
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
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

