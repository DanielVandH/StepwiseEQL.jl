using ..StepwiseEQL
using EpithelialDynamics1D
using OrdinaryDiffEq
using LinearAlgebra
using Bijections
using StatsBase
using ElasticArrays
using DataInterpolations
using BlockDiagonals
const EQL = StepwiseEQL

@testset "Diffusion" begin
    force_law = (δ, p) -> p.k * (p.s - δ)
    force_law_parameters = (k=10.0, s=0.2)
    final_time = 100.0
    damping_constant = 1.0
    initial_condition = [LinRange(0, 15, 16); LinRange(15, 30, 32)] |> unique!
    prob = CellProblem(;
        force_law,
        force_law_parameters,
        final_time,
        damping_constant,
        initial_condition)
    Δt = 0.1
    sol = solve(prob, Tsit5(), saveat=Δt)
    diffusion_basis = BasisSet(
        (u, p) -> inv(u),
        (u, p) -> inv(u^2),
        (u, p) -> inv(u^3)
    )
    diffusion_parameters = nothing

    all_q = Float64[]
    all_∂ₜq = Float64[]
    all_∂ₓq = Float64[]
    all_∂²ₓq² = Float64[]
    all_∂ₓq_bc = Float64[]
    all_dLdt = Float64[]
    for j in 2:length(sol)
        for i in 1:length(initial_condition)
            push!(all_q, EQL.cell_density(sol, i, j))
            push!(all_∂ₜq, EQL.cell_∂q∂t(sol, i, j))
            push!(all_∂ₓq, EQL.cell_∂q∂x(sol, i, j))
            push!(all_∂²ₓq², EQL.cell_∂²q∂x²(sol, i, j))
        end
        push!(all_∂ₓq_bc, EQL.cell_∂q∂x(sol, length(sol.u[j]), j))
        push!(all_dLdt, EQL.cell_dLdt(sol, j))
    end
    min_q, max_q = quantile(all_q, [0.01, 0.99])
    min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [0.05, 0.95])
    min_∂ₓq, max_∂ₓq = quantile(abs.(all_∂ₓq), [0.05, 0.95])
    min_∂²ₓq², max_∂²ₓq² = quantile(abs.(all_∂²ₓq²), [0.05, 0.95])
    min_∂ₓq_bc, max_∂ₓq_bc = quantile(abs.(all_∂ₓq_bc), [0.05, 0.95])
    min_dLdt, max_dLdt = quantile(abs.(all_dLdt), [0.05, 0.95])
    A, b, idx_map, valid_time_indices, qmin, qmax, _sol = EQL.build_basis_system(sol; diffusion_basis, diffusion_parameters, threshold_tol=(q=0.01, dt=0.05, dx=0.05, dx2=0.05, x=0.1))
    @test sol === _sol
    @test qmin ≈ minimum(all_q)
    @test qmax ≈ maximum(all_q)
    @inferred EQL.build_basis_system(sol; diffusion_basis, diffusion_parameters, threshold_tol=(q=0.01, dt=0.05, dx=0.05, dx2=0.05, x=0.1))
    _neqs = length(sol) * length(initial_condition)
    _A = zeros(_neqs, 3)
    _b = zeros(_neqs)
    _idx_map = Bijection{Int,NTuple{3,Int}}()
    global ctr = 1
    for j in 2:length(sol)
        for i in 1:length(initial_condition)
            global ctr
            q = EQL.cell_density(sol, i, j)
            ∂q∂x = EQL.cell_∂q∂x(sol, i, j)
            ∂q∂t = EQL.cell_∂q∂t(sol, i, j)
            ∂²q∂x² = EQL.cell_∂²q∂x²(sol, i, j)
            _A[ctr, 1] = -1 / q^2 * ∂q∂x^2 + 1 / q * ∂²q∂x²
            _A[ctr, 2] = -2 / q^3 * ∂q∂x^2 + 1 / q^2 * ∂²q∂x²
            _A[ctr, 3] = -3 / q^4 * ∂q∂x^2 + 1 / q^3 * ∂²q∂x²
            _b[ctr] = ∂q∂t
            if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq) && (min_∂ₓq ≤ abs(∂q∂x) ≤ max_∂ₓq) && (min_∂²ₓq² ≤ abs(∂²q∂x²) ≤ max_∂²ₓq²) && (0 ≤ sol.u[j][i] ≤ (1 - 0.1) * sol.u[j][end])
                _idx_map[ctr] = (1, i, j)
                ctr += 1
            end
        end
    end
    _A = _A[1:ctr-1, :]
    _b = _b[1:ctr-1]
    @test A ≈ _A
    @test b ≈ _b
    @test idx_map == _idx_map
    @test valid_time_indices == 2:length(sol)

    ## Giving a time range 
    all_q = Float64[]
    all_∂ₜq = Float64[]
    for j in 2:length(sol)
        if 10.0 ≤ sol.t[j] ≤ 85.0
            for i in 1:length(initial_condition)
                push!(all_q, EQL.cell_density(sol, i, j))
                push!(all_∂ₜq, EQL.cell_∂q∂t(sol, i, j))
            end
        end
    end
    min_q, max_q = quantile(all_q, [1e-2, 1 - 1e-2])
    min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [1e-2, 1 - 1e-2])
    A, b, idx_map, time_indices, qmin, qmax, _sol = EQL.build_basis_system(sol; diffusion_basis, diffusion_parameters, threshold_tol=(q=1e-2, dt=1e-2), time_range=(10.0, 85.0))
    @inferred EQL.build_basis_system(sol; diffusion_basis, diffusion_parameters, threshold_tol=(q=1e-2, dt=1e-2), time_range=(10.0, 85.0))
    @test sol === _sol
    @test qmin ≈ minimum(all_q)
    @test qmax ≈ maximum(all_q)
    _neqs = length(sol) * length(initial_condition)
    _A = zeros(_neqs, 3)
    _b = zeros(_neqs)
    _idx_map = Bijection{Int,NTuple{3,Int}}()
    global ctr = 1
    for j in 1:length(sol)
        if 10.0 ≤ sol.t[j] ≤ 85.0
            for i in 1:length(initial_condition)
                global ctr
                q = EQL.cell_density(sol, i, j)
                ∂q∂x = EQL.cell_∂q∂x(sol, i, j)
                ∂q∂t = EQL.cell_∂q∂t(sol, i, j)
                ∂²q∂x² = EQL.cell_∂²q∂x²(sol, i, j)
                _A[ctr, 1] = -1 / q^2 * ∂q∂x^2 + 1 / q * ∂²q∂x²
                _A[ctr, 2] = -2 / q^3 * ∂q∂x^2 + 1 / q^2 * ∂²q∂x²
                _A[ctr, 3] = -3 / q^4 * ∂q∂x^2 + 1 / q^3 * ∂²q∂x²
                _b[ctr] = ∂q∂t
                if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq)
                    _idx_map[ctr] = (1, i, j)
                    ctr += 1
                end
            end
        end
    end
    _A = _A[1:ctr-1, :]
    _b = _b[1:ctr-1]
    @test A ≈ _A
    @test b ≈ _b
    @test idx_map == _idx_map
    @test all(j -> 10.0 ≤ sol.t[j] ≤ 85.0, time_indices)
    @test all(j -> (sol.t[j] < 10.0 || sol.t[j] > 85.0), setdiff(1:length(sol), time_indices))
end

@testset "Proliferation" begin
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
    esol = solve(ens_prob, Tsit5(); trajectories=10, saveat=1.0)
    diffusion_basis = BasisSet(
        (u, k) -> k * inv(u),
        (u, k) -> k * inv(u^2),
        (u, k) -> k * inv(u^3)
    )
    diffusion_parameters = k
    reaction_basis = BasisSet(
        (u, β) -> β * u,
        (u, β) -> β * u^2,
        (u, β) -> β * u^3
    )
    reaction_parameters = Gp.β

    # Aggregated 
    all_q = Float64[]
    all_∂ₜq = Float64[]
    all_∂ₓq = Float64[]
    all_∂²ₓq² = Float64[]
    all_∂ₓq_bc = Float64[]
    all_dLdt = Float64[]
    for k in 1:length(esol)
        sol = esol[k]
        for j in 2:length(sol)
            for i in 1:length(sol[j])
                push!(all_q, EQL.cell_density(sol, i, j))
                push!(all_∂ₜq, EQL.cell_∂q∂t(sol, i, j))
                push!(all_∂ₓq, EQL.cell_∂q∂x(sol, i, j))
                push!(all_∂²ₓq², EQL.cell_∂²q∂x²(sol, i, j))
            end
            push!(all_∂ₓq_bc, EQL.cell_∂q∂x(sol, length(sol[j]), j))
            push!(all_dLdt, EQL.cell_dLdt(sol, j))
        end
    end
    min_q, max_q = quantile(all_q, [0.0, 1])
    min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [1e-4, 1 - 1e-4])
    min_∂ₓq, max_∂ₓq = quantile(abs.(all_∂ₓq), [1e-2, 1 - 1e-2])
    min_∂²ₓq², max_∂²ₓq² = quantile(abs.(all_∂²ₓq²), [0.3, 1 - 0.3])
    min_∂ₓq_bc, max_∂ₓq_bc = quantile(abs.(all_∂ₓq_bc), [1e-2, 1 - 1e-2])
    min_dLdt, max_dLdt = quantile(abs.(all_dLdt), [1e-5, 1 - 1e-5])
    A, b, idx_map, valid_time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
        diffusion_basis, diffusion_parameters, average=Val(false),
        reaction_basis, reaction_parameters, threshold_tol=(q=0.0, dt=1e-4, dx=1e-2, dx2=0.3, dx_bc=1e-2, dL=1e-5))
    @inferred EQL.build_basis_system(esol;
        diffusion_basis, diffusion_parameters, average=Val(false),
        reaction_basis, reaction_parameters, threshold_tol=(q=0.0, dt=1e-4, dx=1e-2, dx2=0.3, dx_bc=1e-2, dL=1e-5))
    @test esol === _sol
    @test qmin ≈ minimum(all_q)
    @test qmax ≈ maximum(all_q)
    _neqs = sum(sum(length.(sol.u)) for sol in esol)
    _A = zeros(_neqs, 6)
    _b = zeros(_neqs)
    _idx_map = Bijection{Int,NTuple{3,Int}}()
    global ctr = 1
    for k in 1:length(esol)
        for j in 2:length(esol[k])
            for i in 1:length(esol[k][j])
                global ctr
                q = EQL.cell_density(esol[k], i, j)
                ∂q∂x = EQL.cell_∂q∂x(esol[k], i, j)
                ∂q∂t = EQL.cell_∂q∂t(esol[k], i, j)
                ∂²q∂x² = EQL.cell_∂²q∂x²(esol[k], i, j)
                _A[ctr, 1] = -diffusion_parameters / q^2 * ∂q∂x^2 + diffusion_parameters / q * ∂²q∂x²
                _A[ctr, 2] = -2diffusion_parameters / q^3 * ∂q∂x^2 + diffusion_parameters / q^2 * ∂²q∂x²
                _A[ctr, 3] = -3diffusion_parameters / q^4 * ∂q∂x^2 + diffusion_parameters / q^3 * ∂²q∂x²
                _A[ctr, 4] = Gp.β * q
                _A[ctr, 5] = Gp.β * q^2
                _A[ctr, 6] = Gp.β * q^3
                _b[ctr] = ∂q∂t
                if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq) && (min_∂ₓq ≤ abs(∂q∂x) ≤ max_∂ₓq) && (min_∂²ₓq² ≤ abs(∂²q∂x²) ≤ max_∂²ₓq²)
                    _idx_map[ctr] = (k, i, j)
                    ctr += 1
                end
            end
        end
    end
    _A = _A[1:ctr-1, :]
    _b = _b[1:ctr-1]
    @test A ≈ _A
    @test b ≈ _b
    @test idx_map == _idx_map
    @test valid_time_indices == 2:length(esol[1])

    # Fixing diffusion parameters
    min_q, max_q = quantile(all_q, [0.1, 0.9])
    min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [0.2, 0.8])
    A, b, idx_map, valid_time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol; average=Val(false),
        diffusion_basis, diffusion_parameters, diffusion_theta=[0.1, 1.1, -0.2],
        reaction_basis, reaction_parameters, threshold_tol=(q=0.1, dt=0.2))
    @inferred EQL.build_basis_system(esol; average=Val(false),
        diffusion_basis, diffusion_parameters, diffusion_theta=[0.1, 1.1, -0.2],
        reaction_basis, reaction_parameters, threshold_tol=(q=0.1, dt=0.2))
    @test esol === _sol
    _neqs = sum(sum(length.(sol.u)) for sol in esol)
    _A = zeros(_neqs, 3)
    _b = zeros(_neqs)
    _idx_map = Bijection{Int,NTuple{3,Int}}()
    global ctr = 1
    for k in 1:length(esol)
        for j in 2:length(esol[k])
            for i in 1:length(esol[k][j])
                global ctr
                q = EQL.cell_density(esol[k], i, j)
                ∂q∂x = EQL.cell_∂q∂x(esol[k], i, j)
                ∂q∂t = EQL.cell_∂q∂t(esol[k], i, j)
                ∂²q∂x² = EQL.cell_∂²q∂x²(esol[k], i, j)
                _A[ctr, 1] = Gp.β * q
                _A[ctr, 2] = Gp.β * q^2
                _A[ctr, 3] = Gp.β * q^3
                Dq = 0.1diffusion_parameters / q + 1.1diffusion_parameters / q^2 - 0.2diffusion_parameters / q^3
                D′q = -0.1diffusion_parameters / q^2 - 2.2diffusion_parameters / q^3 + 0.6diffusion_parameters / q^4
                _b[ctr] = ∂q∂t - D′q * ∂q∂x^2 - Dq * ∂²q∂x²
                if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq)
                    _idx_map[ctr] = (k, i, j)
                    ctr += 1
                end
            end
        end
    end
    _A = _A[1:ctr-1, :]
    _b = _b[1:ctr-1]
    @test A ≈ _A
    @test b ≈ _b
    @test idx_map == _idx_map
    @test valid_time_indices == 2:length(esol[1])

    # Fixing reaction parameters
    min_q, max_q = quantile(all_q, [0.01, 1 - 0.01])
    min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [0.1, 0.9])
    A, b, idx_map, valid_time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
        diffusion_basis, diffusion_parameters, average=Val(false),
        reaction_basis, reaction_parameters, reaction_theta=[0.5, -0.7, 1.3],
        threshold_tol=(q=0.01, dt=0.1))
    @inferred EQL.build_basis_system(esol;
        diffusion_basis, diffusion_parameters, average=Val(false),
        reaction_basis, reaction_parameters, reaction_theta=[0.5, -0.7, 1.3],
        threshold_tol=(q=0.01, dt=0.1))
    @test esol === _sol
    _neqs = sum(sum(length.(sol.u)) for sol in esol)
    _A = zeros(_neqs, 3)
    _b = zeros(_neqs)
    _idx_map = Bijection{Int,NTuple{3,Int}}()
    global ctr = 1
    for k in 1:length(esol)
        for j in 2:length(esol[k])
            for i in 1:length(esol[k][j])
                global ctr
                q = EQL.cell_density(esol[k], i, j)
                ∂q∂x = EQL.cell_∂q∂x(esol[k], i, j)
                ∂q∂t = EQL.cell_∂q∂t(esol[k], i, j)
                ∂²q∂x² = EQL.cell_∂²q∂x²(esol[k], i, j)
                _A[ctr, 1] = -diffusion_parameters / q^2 * ∂q∂x^2 + diffusion_parameters / q * ∂²q∂x²
                _A[ctr, 2] = -2diffusion_parameters / q^3 * ∂q∂x^2 + diffusion_parameters / q^2 * ∂²q∂x²
                _A[ctr, 3] = -3diffusion_parameters / q^4 * ∂q∂x^2 + diffusion_parameters / q^3 * ∂²q∂x²
                _b[ctr] = ∂q∂t - 0.5Gp.β * q + 0.7Gp.β * q^2 - 1.3Gp.β * q^3
                if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq)
                    _idx_map[ctr] = (k, i, j)
                    ctr += 1
                end
            end
        end
    end
    _A = _A[1:ctr-1, :]
    _b = _b[1:ctr-1]
    @test A ≈ _A
    @test b ≈ _b
    @test idx_map == _idx_map
    @test valid_time_indices == 2:length(esol[1])

    # Giving a time range 
    all_q = Float64[]
    all_∂ₜq = Float64[]
    for k in 1:length(esol)
        sol = esol[k]
        for j in 2:length(sol)
            for i in 1:length(sol[j])
                if 7.5 ≤ sol.t[j] ≤ 27.3
                    push!(all_q, EQL.cell_density(sol, i, j))
                    push!(all_∂ₜq, EQL.cell_∂q∂t(sol, i, j))
                end
            end
        end
    end
    min_q, max_q = quantile(all_q, [0.1, 0.9])
    min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [0.2, 0.8])
    A, b, idx_map, time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
        diffusion_basis, diffusion_parameters, average=Val(false),
        reaction_basis, reaction_parameters, reaction_theta=[0.5, -0.7, 1.3],
        threshold_tol=(q=0.1, dt=0.2),
        time_range=(7.5, 27.3))
    @inferred EQL.build_basis_system(esol;
        diffusion_basis, diffusion_parameters,
        reaction_basis, reaction_parameters, reaction_theta=[0.5, -0.7, 1.3],
        threshold_tol=(q=0.1, dt=0.1), average=Val(false),
        time_range=(7.5, 27.3))
    @test _sol === esol
    _neqs = sum(sum(length.(sol.u)) for sol in esol)
    _A = zeros(_neqs, 3)
    _b = zeros(_neqs)
    _idx_map = Bijection{Int,NTuple{3,Int}}()
    global ctr = 1
    for k in 1:length(esol)
        for j in 1:length(esol[k])
            if 7.5 ≤ esol[k].t[j] ≤ 27.3
                for i in 1:length(esol[k][j])
                    global ctr
                    q = EQL.cell_density(esol[k], i, j)
                    ∂q∂x = EQL.cell_∂q∂x(esol[k], i, j)
                    ∂q∂t = EQL.cell_∂q∂t(esol[k], i, j)
                    ∂²q∂x² = EQL.cell_∂²q∂x²(esol[k], i, j)
                    _A[ctr, 1] = -diffusion_parameters / q^2 * ∂q∂x^2 + diffusion_parameters / q * ∂²q∂x²
                    _A[ctr, 2] = -2diffusion_parameters / q^3 * ∂q∂x^2 + diffusion_parameters / q^2 * ∂²q∂x²
                    _A[ctr, 3] = -3diffusion_parameters / q^4 * ∂q∂x^2 + diffusion_parameters / q^3 * ∂²q∂x²
                    _b[ctr] = ∂q∂t - 0.5Gp.β * q + 0.7Gp.β * q^2 - 1.3Gp.β * q^3
                    if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq)
                        _idx_map[ctr] = (k, i, j)
                        ctr += 1
                    end
                end
            end
        end
    end
    _A = _A[1:ctr-1, :]
    _b = _b[1:ctr-1]
    @test A ≈ _A
    @test b ≈ _b
    @test idx_map == _idx_map
    @test all(j -> 7.5 ≤ esol[1].t[j] ≤ 27.3, time_indices)
    @test all(j -> (esol[1].t[j] < 7.5 || esol[1].t[j] > 27.3), setdiff(1:length(esol[1]), time_indices))

    # Averaging
    indices = rand(eachindex(esol), 20)
    asol = EQL.AveragedODESolution(esol, 400, indices)
    all_q = Float64[]
    all_∂ₜq = Float64[]
    for j in 2:length(asol)
        for i in 1:length(asol.q[j])
            if 7.5 ≤ asol.t[j] ≤ 27.3
                push!(all_q, EQL.cell_density(asol, i, j))
                push!(all_∂ₜq, EQL.cell_∂q∂t(asol, i, j))
            end
        end
    end
    min_q, max_q = quantile(all_q, [0.1, 0.9])
    min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [0.2, 0.8])
    A, b, idx_map, time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
        diffusion_basis, diffusion_parameters,
        reaction_basis, reaction_parameters, reaction_theta=[0.5, -0.7, 1.3],
        threshold_tol=(q=0.1, dt=0.2),
        time_range=(q=7.5, dt=27.3),
        simulation_indices=indices,
        num_knots=400)
    @test asol.q == _sol.q
    @test asol.u == _sol.u
    @test asol.t == _sol.t
    @test asol.cell_sol === _sol.cell_sol
    _neqs = sum(length.(asol.q))
    _A = zeros(_neqs, 3)
    _b = zeros(_neqs)
    _idx_map = Bijection{Int,NTuple{3,Int}}()
    global ctr = 1
    for j in 1:length(asol)
        if 7.5 ≤ asol.t[j] ≤ 27.3
            for i in 1:length(asol.q[j])
                global ctr
                q = EQL.cell_density(asol, i, j)
                ∂q∂x = EQL.cell_∂q∂x(asol, i, j)
                ∂q∂t = EQL.cell_∂q∂t(asol, i, j)
                ∂²q∂x² = EQL.cell_∂²q∂x²(asol, i, j)
                _A[ctr, 1] = -diffusion_parameters / q^2 * ∂q∂x^2 + diffusion_parameters / q * ∂²q∂x²
                _A[ctr, 2] = -2diffusion_parameters / q^3 * ∂q∂x^2 + diffusion_parameters / q^2 * ∂²q∂x²
                _A[ctr, 3] = -3diffusion_parameters / q^4 * ∂q∂x^2 + diffusion_parameters / q^3 * ∂²q∂x²
                _b[ctr] = ∂q∂t - 0.5Gp.β * q + 0.7Gp.β * q^2 - 1.3Gp.β * q^3
                if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq)
                    _idx_map[ctr] = (1, i, j)
                    ctr += 1
                end
            end
        end
    end
    _A = _A[1:ctr-1, :]
    _b = _b[1:ctr-1]
    @test A ≈ _A
    @test b ≈ _b
    @test idx_map == _idx_map
    @test all(j -> 7.5 ≤ asol.t[j] ≤ 27.3, time_indices)
    @test all(j -> (asol.t[j] < 7.5 || asol.t[j] > 27.3), setdiff(1:length(asol), time_indices))

    # Test the averaged defaults
    indices = eachindex(esol)
    asol = EQL.AveragedODESolution(esol, 100, indices)
    all_q = Float64[]
    all_∂ₜq = Float64[]
    for j in 2:length(asol)
        for i in 1:length(asol.q[j])
            push!(all_q, EQL.cell_density(asol, i, j))
            push!(all_∂ₜq, EQL.cell_∂q∂t(asol, i, j))
        end
    end
    min_q, max_q = quantile(all_q, [0.1, 0.9])
    min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [0.2, 0.8])
    A, b, idx_map, time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
        diffusion_basis, diffusion_parameters,
        reaction_basis, reaction_parameters,
        threshold_tol=(q=0.1, dt=0.2))
    @test asol.q == _sol.q
    @test asol.u == _sol.u
    @test asol.t == _sol.t
    @test asol.cell_sol === _sol.cell_sol
    _neqs = sum(length.(asol.q))
    _A = zeros(_neqs, 6)
    _b = zeros(_neqs)
    _idx_map = Bijection{Int,NTuple{3,Int}}()
    global ctr = 1
    for j in 2:length(asol)
        for i in 1:length(asol.q[j])
            global ctr
            q = EQL.cell_density(asol, i, j)
            ∂q∂x = EQL.cell_∂q∂x(asol, i, j)
            ∂q∂t = EQL.cell_∂q∂t(asol, i, j)
            ∂²q∂x² = EQL.cell_∂²q∂x²(asol, i, j)
            _A[ctr, 1] = -diffusion_parameters / q^2 * ∂q∂x^2 + diffusion_parameters / q * ∂²q∂x²
            _A[ctr, 2] = -2diffusion_parameters / q^3 * ∂q∂x^2 + diffusion_parameters / q^2 * ∂²q∂x²
            _A[ctr, 3] = -3diffusion_parameters / q^4 * ∂q∂x^2 + diffusion_parameters / q^3 * ∂²q∂x²
            _A[ctr, 4] = Gp.β * q
            _A[ctr, 5] = Gp.β * q^2
            _A[ctr, 6] = Gp.β * q^3
            _b[ctr] = ∂q∂t
            if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq)
                _idx_map[ctr] = (1, i, j)
                ctr += 1
            end
        end
    end
    _A = _A[1:ctr-1, :]
    _b = _b[1:ctr-1]
    @test A ≈ _A
    @test b ≈ _b
    @test idx_map == _idx_map
    @test all(j -> 0.0 ≤ asol.t[j] ≤ final_time, time_indices)

    # Providing a subset of simulation indices for aggregating 
    indices = sample(eachindex(esol), length(esol) ÷ 2, replace=false)
    all_q = Float64[]
    all_∂ₜq = Float64[]
    for k in indices
        sol = esol[k]
        for j in 2:length(sol)
            for i in 1:length(sol[j])
                push!(all_q, EQL.cell_density(sol, i, j))
                push!(all_∂ₜq, EQL.cell_∂q∂t(sol, i, j))
            end
        end
    end
    min_q, max_q = quantile(all_q, [0.0, 1])
    min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [1e-4, 1 - 1e-4])
    A, b, idx_map, valid_time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
        diffusion_basis, diffusion_parameters, average=Val(false),
        reaction_basis, reaction_parameters, threshold_tol=(dt=1e-4,),
        simulation_indices=indices)
    @test esol === _sol
    _neqs = sum(sum(length.(esol[k].u)) for k in indices)
    _A = zeros(_neqs, 6)
    _b = zeros(_neqs)
    _idx_map = Bijection{Int,NTuple{3,Int}}()
    global ctr = 1
    for k in indices
        for j in 2:length(esol[k])
            for i in 1:length(esol[k][j])
                global ctr
                q = EQL.cell_density(esol[k], i, j)
                ∂q∂x = EQL.cell_∂q∂x(esol[k], i, j)
                ∂q∂t = EQL.cell_∂q∂t(esol[k], i, j)
                ∂²q∂x² = EQL.cell_∂²q∂x²(esol[k], i, j)
                _A[ctr, 1] = -diffusion_parameters / q^2 * ∂q∂x^2 + diffusion_parameters / q * ∂²q∂x²
                _A[ctr, 2] = -2diffusion_parameters / q^3 * ∂q∂x^2 + diffusion_parameters / q^2 * ∂²q∂x²
                _A[ctr, 3] = -3diffusion_parameters / q^4 * ∂q∂x^2 + diffusion_parameters / q^3 * ∂²q∂x²
                _A[ctr, 4] = Gp.β * q
                _A[ctr, 5] = Gp.β * q^2
                _A[ctr, 6] = Gp.β * q^3
                _b[ctr] = ∂q∂t
                if (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq)
                    _idx_map[ctr] = (k, i, j)
                    ctr += 1
                end
            end
        end
    end
    _A = _A[1:ctr-1, :]
    _b = _b[1:ctr-1]
    @test A ≈ _A
    @test b ≈ _b
    @test idx_map == _idx_map
    @test valid_time_indices == 2:length(esol[1])

    # Averaging with a CubicSpline
    indices = rand(eachindex(esol), 20)
    asol = EQL.AveragedODESolution(esol, 400, indices, CubicSpline)
    all_q = Float64[]
    all_∂ₜq = Float64[]
    for j in 2:length(asol)
        for i in 1:length(asol.q[j])
            if 7.5 ≤ asol.t[j] ≤ 27.3
                push!(all_q, EQL.cell_density(asol, i, j))
                push!(all_∂ₜq, EQL.cell_∂q∂t(asol, i, j))
            end
        end
    end
    min_q, max_q = quantile(all_q, [0.1, 0.9])
    min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [0.2, 0.8])
    A, b, idx_map, time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
        diffusion_basis, diffusion_parameters,
        reaction_basis, reaction_parameters, reaction_theta=[0.5, -0.7, 1.3],
        threshold_tol=(q=0.1, dt=0.2),
        time_range=(7.5, 27.3),
        simulation_indices=indices,
        num_knots=400,
        avg_interp_fnc=CubicSpline)
    @test asol.q == _sol.q
    @test asol.u == _sol.u
    @test asol.t == _sol.t
    @test asol.cell_sol === _sol.cell_sol
    _neqs = sum(length.(asol.q))
    _A = zeros(_neqs, 3)
    _b = zeros(_neqs)
    _idx_map = Bijection{Int,NTuple{3,Int}}()
    global ctr = 1
    for j in 1:length(asol)
        if 7.5 ≤ asol.t[j] ≤ 27.3
            for i in 1:length(asol.q[j])
                global ctr
                q = EQL.cell_density(asol, i, j)
                ∂q∂x = EQL.cell_∂q∂x(asol, i, j)
                ∂q∂t = EQL.cell_∂q∂t(asol, i, j)
                ∂²q∂x² = EQL.cell_∂²q∂x²(asol, i, j)
                _A[ctr, 1] = -diffusion_parameters / q^2 * ∂q∂x^2 + diffusion_parameters / q * ∂²q∂x²
                _A[ctr, 2] = -2diffusion_parameters / q^3 * ∂q∂x^2 + diffusion_parameters / q^2 * ∂²q∂x²
                _A[ctr, 3] = -3diffusion_parameters / q^4 * ∂q∂x^2 + diffusion_parameters / q^3 * ∂²q∂x²
                _b[ctr] = ∂q∂t - 0.5Gp.β * q + 0.7Gp.β * q^2 - 1.3Gp.β * q^3
                if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq)
                    _idx_map[ctr] = (1, i, j)
                    ctr += 1
                end
            end
        end
    end
    _A = _A[1:ctr-1, :]
    _b = _b[1:ctr-1]
    @test A ≈ _A
    @test b ≈ _b
    @test idx_map == _idx_map
    @test all(j -> 7.5 ≤ asol.t[j] ≤ 27.3, time_indices)
    @test all(j -> (asol.t[j] < 7.5 || asol.t[j] > 27.3), setdiff(1:length(asol), time_indices))
end

@testset "Moving Boundary without Proliferation" begin
    force_law = (δ, p) -> p.k * (p.s - δ)
    force_law_parameters = (k=10.0, s=0.2)
    final_time = 100.0
    damping_constant = 1.0
    initial_condition = [LinRange(0, 15, 16); LinRange(15, 30, 32)] |> unique!
    prob = CellProblem(;
        force_law,
        force_law_parameters,
        final_time,
        damping_constant,
        initial_condition,
        fix_right=false)
    Δt = 0.1
    sol = solve(prob, Tsit5(), saveat=Δt)
    diffusion_basis = BasisSet(
        (u, p) -> inv(u),
        (u, p) -> inv(u^2),
        (u, p) -> inv(u^3)
    )
    diffusion_parameters = nothing
    rhs_basis = BasisSet(
        (u, p) -> u^2,
        (u, p) -> u^3
    )
    rhs_parameters = nothing
    moving_boundary_basis = BasisSet(
        (u, p) -> inv(u),
        (u, p) -> inv(u^2),
        (u, p) -> inv(u^3)
    )
    moving_boundary_parameters = nothing

    all_q = Float64[]
    all_∂ₜq = Float64[]
    all_∂ₓq = Float64[]
    all_∂²ₓq² = Float64[]
    all_∂ₓq_bc = Float64[]
    all_dLdt = Float64[]
    for j in 2:length(sol)
        for i in 1:length(initial_condition)
            push!(all_q, EQL.cell_density(sol, i, j))
            push!(all_∂ₜq, EQL.cell_∂q∂t(sol, i, j))
            push!(all_∂ₓq, EQL.cell_∂q∂x(sol, i, j))
            push!(all_∂²ₓq², EQL.cell_∂²q∂x²(sol, i, j))
        end
        push!(all_∂ₓq_bc, EQL.cell_∂q∂x(sol, length(sol.u[j]), j))
        push!(all_dLdt, EQL.cell_dLdt(sol, j))
    end
    min_q, max_q = quantile(all_q, [0.01, 0.99])
    min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [0.05, 0.95])
    min_∂ₓq, max_∂ₓq = quantile(abs.(all_∂ₓq), [0.05, 0.95])
    min_∂²ₓq², max_∂²ₓq² = quantile(abs.(all_∂²ₓq²), [0.05, 0.95])
    min_∂ₓq_bc, max_∂ₓq_bc = quantile(abs.(all_∂ₓq_bc), [0.01, 0.99])
    min_dLdt, max_dLdt = quantile(abs.(all_dLdt), [0.02, 0.98])
    A, b, idx_map, valid_time_indices, qmin, qmax, _sol = EQL.build_basis_system(sol; diffusion_basis, diffusion_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters,
        threshold_tol=(q=0.01, dt=0.05, dx=0.05, dx2=0.05, dx_bc=0.01, dL=0.02, x=0.01))
    @test sol === _sol
    @test qmin == minimum(all_q)
    @test qmax == maximum(all_q)
    @inferred EQL.build_basis_system(sol; diffusion_basis, diffusion_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters,
        threshold_tol=(q=0.01, dt=0.05, dx=0.05, dx2=0.05, dx_bc=0.01, dL=0.02, x=0.01))
    _neqs = length(sol) * length(initial_condition)
    _A1 = zeros(_neqs, 3)
    _b1 = zeros(_neqs)
    _idx_map = Bijection{Int,NTuple{3,Int}}()
    global ctr = 1
    for j in 2:length(sol)
        for i in 1:length(initial_condition)
            global ctr
            q = EQL.cell_density(sol, i, j)
            ∂q∂x = EQL.cell_∂q∂x(sol, i, j)
            ∂q∂t = EQL.cell_∂q∂t(sol, i, j)
            ∂²q∂x² = EQL.cell_∂²q∂x²(sol, i, j)
            _A1[ctr, 1] = -1 / q^2 * ∂q∂x^2 + 1 / q * ∂²q∂x²
            _A1[ctr, 2] = -2 / q^3 * ∂q∂x^2 + 1 / q^2 * ∂²q∂x²
            _A1[ctr, 3] = -3 / q^4 * ∂q∂x^2 + 1 / q^3 * ∂²q∂x²
            _b1[ctr] = ∂q∂t
            if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq) && (min_∂ₓq ≤ abs(∂q∂x) ≤ max_∂ₓq) && (min_∂²ₓq² ≤ abs(∂²q∂x²) ≤ max_∂²ₓq²) && (0 ≤ sol.u[j][i] ≤ (1 - 0.01) * sol.u[j][end])
                _idx_map[ctr] = (1, i, j)
                ctr += 1
            end
        end
    end
    _A1 = _A1[1:ctr-1, :]
    _b1 = _b1[1:ctr-1]
    _A2 = zeros(length(sol), 2)
    _b2 = zeros(length(sol))
    global ctr1 = 1
    for j in 2:length(sol)
        global ctr1
        global ctr
        q = EQL.cell_density(sol, length(sol.u[j]), j)
        ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
        dLdt = EQL.cell_dLdt(sol, j)
        _A2[ctr1, 1] = q^2
        _A2[ctr1, 2] = q^3
        _b2[ctr1] = ∂q∂x
        if (min_∂ₓq_bc ≤ abs(∂q∂x) ≤ max_∂ₓq_bc)
            _idx_map[ctr] = (1, 0, j)
            ctr1 += 1
            ctr += 1
        end
    end
    _A2 = _A2[1:ctr1-1, :]
    _b2 = _b2[1:ctr1-1]
    _A3 = zeros(length(sol), 3)
    _b3 = zeros(length(sol))
    global ctr2 = 1
    for j in 2:length(sol)
        global ctr2
        global ctr
        q = EQL.cell_density(sol, length(sol.u[j]), j)
        ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
        dLdt = EQL.cell_dLdt(sol, j)
        _A3[ctr2, 1] = -1 / q^1 * ∂q∂x
        _A3[ctr2, 2] = -1 / q^2 * ∂q∂x
        _A3[ctr2, 3] = -1 / q^3 * ∂q∂x
        _b3[ctr2] = q * dLdt
        if (min_dLdt ≤ abs(dLdt) ≤ max_dLdt)
            _idx_map[ctr] = (1, -1, j)
            ctr2 += 1
            ctr += 1
        end
    end
    _A3 = _A3[1:ctr2-1, :]
    _b3 = _b3[1:ctr2-1]
    _A = (Matrix ∘ BlockDiagonal)([_A1, _A2, _A3])
    _b = vcat(_b1, _b2, _b3)
    @test A ≈ _A
    @test b ≈ _b
    @test idx_map == _idx_map
    @test valid_time_indices == 2:length(sol)

    ## Giving a time range 
    A, b, idx_map, time_indices, qmin, qmax, _sol = EQL.build_basis_system(sol; diffusion_basis, time_range=(27.0, 83.0), diffusion_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters)
    @test sol === _sol
    @inferred EQL.build_basis_system(sol; diffusion_basis, diffusion_parameters, time_range=(27.0, 83.0), rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters)
    _neqs = length(sol) * length(initial_condition)
    _A1 = zeros(_neqs, 3)
    _b1 = zeros(_neqs)
    _idx_map = Bijection{Int,NTuple{3,Int}}()
    global ctr = 1
    for j in 2:length(sol)
        if 27.0 ≤ sol.t[j] ≤ 83.0
            for i in 1:length(initial_condition)
                global ctr
                q = EQL.cell_density(sol, i, j)
                ∂q∂x = EQL.cell_∂q∂x(sol, i, j)
                ∂q∂t = EQL.cell_∂q∂t(sol, i, j)
                ∂²q∂x² = EQL.cell_∂²q∂x²(sol, i, j)
                _A1[ctr, 1] = -1 / q^2 * ∂q∂x^2 + 1 / q * ∂²q∂x²
                _A1[ctr, 2] = -2 / q^3 * ∂q∂x^2 + 1 / q^2 * ∂²q∂x²
                _A1[ctr, 3] = -3 / q^4 * ∂q∂x^2 + 1 / q^3 * ∂²q∂x²
                _b1[ctr] = ∂q∂t
                _idx_map[ctr] = (1, i, j)
                ctr += 1
            end
        end
    end
    _A1 = _A1[1:ctr-1, :]
    _b1 = _b1[1:ctr-1]
    _A2 = zeros(length(sol), 2)
    _b2 = zeros(length(sol))
    global ctr = 1
    for j in 2:length(sol)
        if 27.0 ≤ sol.t[j] ≤ 83.0
            global ctr
            q = EQL.cell_density(sol, length(sol.u[j]), j)
            ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
            dLdt = EQL.cell_dLdt(sol, j)
            _A2[ctr, 1] = q^2
            _A2[ctr, 2] = q^3
            _b2[ctr] = ∂q∂x
            _idx_map[size(_A1, 1)+ctr] = (1, 0, j)
            ctr += 1
        end
    end
    _A2 = _A2[1:ctr-1, :]
    _b2 = _b2[1:ctr-1]
    _A3 = zeros(length(sol), 3)
    _b3 = zeros(length(sol))
    global ctr = 1
    for j in 2:length(sol)
        if 27.0 ≤ sol.t[j] ≤ 83.0
            global ctr
            q = EQL.cell_density(sol, length(sol.u[j]), j)
            ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
            dLdt = EQL.cell_dLdt(sol, j)
            _A3[ctr, 1] = -1 / q * ∂q∂x
            _A3[ctr, 2] = -1 / q^2 * ∂q∂x
            _A3[ctr, 3] = -1 / q^3 * ∂q∂x
            _b3[ctr] = q * dLdt
            _idx_map[size(_A1, 1)+size(_A2, 1)+ctr] = (1, -1, j)
            ctr += 1
        end
    end
    _A3 = _A3[1:ctr-1, :]
    _b3 = _b3[1:ctr-1]
    _A = (Matrix ∘ BlockDiagonal)([_A1, _A2, _A3])
    _b = vcat(_b1, _b2, _b3)
    @test A ≈ _A
    @test b ≈ _b
    @test idx_map == _idx_map
    @test all(j -> 27.0 ≤ sol.t[j] ≤ 83.0, time_indices)
    @test all(j -> (sol.t[j] < 27.0 || sol.t[j] > 83.0), setdiff(1:length(sol), time_indices))
end

@testset "Moving Boundary with Proliferation" begin
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
        proliferation_law_parameters=Gp,
        fix_right=false)
    ens_prob = EnsembleProblem(prob)
    esol = solve(ens_prob, Tsit5(); trajectories=10, saveat=1.0)
    diffusion_basis = BasisSet(
        (u, k) -> k * inv(u),
        (u, k) -> k * inv(u^2),
        (u, k) -> k * inv(u^3)
    )
    diffusion_parameters = k
    reaction_basis = BasisSet(
        (u, β) -> β * u,
        (u, β) -> β * u^2,
        (u, β) -> β * u^3
    )
    reaction_parameters = Gp.β
    rhs_basis = BasisSet(
        (u, p) -> u^2,
        (u, p) -> u^3
    )
    rhs_parameters = nothing
    moving_boundary_basis = BasisSet(
        (u, p) -> inv(u),
        (u, p) -> inv(u^2),
        (u, p) -> inv(u^3)
    )
    moving_boundary_parameters = nothing

    # Aggregated 
    begin
        all_q = Float64[]
        all_∂ₜq = Float64[]
        all_∂ₓq = Float64[]
        all_∂²ₓq² = Float64[]
        all_∂ₓq_bc = Float64[]
        all_dLdt = Float64[]
        for k in 1:length(esol)
            sol = esol[k]
            for j in 2:length(sol)
                for i in 1:length(sol[j])
                    push!(all_q, EQL.cell_density(sol, i, j))
                    push!(all_∂ₜq, EQL.cell_∂q∂t(sol, i, j))
                    push!(all_∂ₓq, EQL.cell_∂q∂x(sol, i, j))
                    push!(all_∂²ₓq², EQL.cell_∂²q∂x²(sol, i, j))
                end
                push!(all_∂ₓq_bc, EQL.cell_∂q∂x(sol, length(sol[j]), j))
                push!(all_dLdt, EQL.cell_dLdt(sol, j))
            end
        end
        min_q, max_q = quantile(all_q, [0.0, 1])
        min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [1e-4, 1 - 1e-4])
        min_∂ₓq, max_∂ₓq = quantile(abs.(all_∂ₓq), [1e-2, 1 - 1e-2])
        min_∂²ₓq², max_∂²ₓq² = quantile(abs.(all_∂²ₓq²), [0.3, 1 - 0.3])
        min_∂ₓq_bc, max_∂ₓq_bc = quantile(abs.(all_∂ₓq_bc), [1e-2, 1 - 1e-2])
        min_dLdt, max_dLdt = quantile(abs.(all_dLdt), [1e-5, 1 - 1e-5])
        A, b, idx_map, valid_time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
            diffusion_basis, diffusion_parameters,
            reaction_basis, reaction_parameters,
            rhs_basis, rhs_parameters, average=Val(false),
            moving_boundary_basis, moving_boundary_parameters,
            threshold_tol=(q=0.0, dt=1e-4, dx=1e-2, dx2=0.3, dx_bc=1e-2, dL=1e-5))
        @test esol === _sol
        @test qmin == minimum(all_q)
        @test qmax == maximum(all_q)
        _neqs = sum(sum(length.(sol.u)) for sol in esol)
        _A1 = zeros(_neqs, 6)
        _b1 = zeros(_neqs)
        _idx_map = Bijection{Int,NTuple{3,Int}}()
        global ctr = 1
        for k in 1:length(esol)
            for j in 2:length(esol[k])
                for i in 1:length(esol[k][j])
                    global ctr
                    q = EQL.cell_density(esol[k], i, j)
                    ∂q∂x = EQL.cell_∂q∂x(esol[k], i, j)
                    ∂q∂t = EQL.cell_∂q∂t(esol[k], i, j)
                    ∂²q∂x² = EQL.cell_∂²q∂x²(esol[k], i, j)
                    _A1[ctr, 1] = -diffusion_parameters / q^2 * ∂q∂x^2 + diffusion_parameters / q * ∂²q∂x²
                    _A1[ctr, 2] = -2diffusion_parameters / q^3 * ∂q∂x^2 + diffusion_parameters / q^2 * ∂²q∂x²
                    _A1[ctr, 3] = -3diffusion_parameters / q^4 * ∂q∂x^2 + diffusion_parameters / q^3 * ∂²q∂x²
                    _A1[ctr, 4] = Gp.β * q
                    _A1[ctr, 5] = Gp.β * q^2
                    _A1[ctr, 6] = Gp.β * q^3
                    _b1[ctr] = ∂q∂t
                    if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq) && (min_∂ₓq ≤ abs(∂q∂x) ≤ max_∂ₓq) && (min_∂²ₓq² ≤ abs(∂²q∂x²) ≤ max_∂²ₓq²)
                        _idx_map[ctr] = (k, i, j)
                        ctr += 1
                    end
                end
            end
        end
        _A1 = _A1[1:ctr-1, :]
        _b1 = _b1[1:ctr-1]
        _A2 = zeros(sum(length.(esol.u)), 2)
        _b2 = zeros(sum(length.(esol.u)))
        global ctr = 1
        for k in 1:length(esol)
            sol = esol[k]
            for j in 2:length(sol)
                global ctr
                q = EQL.cell_density(sol, length(sol.u[j]), j)
                ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
                dLdt = EQL.cell_dLdt(sol, j)
                _A2[ctr, 1] = q^2
                _A2[ctr, 2] = q^3
                _b2[ctr] = ∂q∂x
                if (min_∂ₓq_bc ≤ abs(∂q∂x) ≤ max_∂ₓq_bc)
                    _idx_map[size(_A1, 1)+ctr] = (k, 0, j)
                    ctr += 1
                end
            end
        end
        _A2 = _A2[1:ctr-1, :]
        _b2 = _b2[1:ctr-1]
        _A3 = zeros(sum(length.(esol.u)), 3)
        _b3 = zeros(sum(length.(esol.u)))
        global ctr = 1
        for k in 1:length(esol)
            sol = esol[k]
            for j in 2:length(sol)
                global ctr
                q = EQL.cell_density(sol, length(sol.u[j]), j)
                ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
                dLdt = EQL.cell_dLdt(sol, j)
                _A3[ctr, 1] = -1 / q * ∂q∂x
                _A3[ctr, 2] = -1 / q^2 * ∂q∂x
                _A3[ctr, 3] = -1 / q^3 * ∂q∂x
                _b3[ctr] = q * dLdt
                if (min_dLdt ≤ abs(dLdt) ≤ max_dLdt)
                    _idx_map[size(_A1, 1)+size(_A2, 1)+ctr] = (k, -1, j)
                    ctr += 1
                end
            end
        end
        _A3 = _A3[1:ctr-1, :]
        _b3 = _b3[1:ctr-1]
        _A = (Matrix ∘ BlockDiagonal)([_A1, _A2, _A3])
        _b = vcat(_b1, _b2, _b3)
        @test A ≈ _A
        @test b ≈ _b
        @test idx_map == _idx_map
        @test valid_time_indices == 2:length(esol[1])
    end

    # Fixing diffusion parameters
    begin
        A, b, idx_map, valid_time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
            diffusion_basis, diffusion_parameters, diffusion_theta=[0.1, 1.1, -0.2],
            reaction_basis, reaction_parameters,
            rhs_basis, rhs_parameters, average=Val(false),
            moving_boundary_basis, moving_boundary_parameters)
        @test esol === _sol
        _neqs = sum(sum(length.(sol.u)) for sol in esol)
        _A1 = zeros(_neqs, 3)
        _b1 = zeros(_neqs)
        _idx_map = Bijection{Int,NTuple{3,Int}}()
        global ctr = 1
        for k in 1:length(esol)
            for j in 2:length(esol[k])
                for i in 1:length(esol[k][j])
                    global ctr
                    q = EQL.cell_density(esol[k], i, j)
                    ∂q∂x = EQL.cell_∂q∂x(esol[k], i, j)
                    ∂q∂t = EQL.cell_∂q∂t(esol[k], i, j)
                    ∂²q∂x² = EQL.cell_∂²q∂x²(esol[k], i, j)
                    _A1[ctr, 1] = Gp.β * q
                    _A1[ctr, 2] = Gp.β * q^2
                    _A1[ctr, 3] = Gp.β * q^3
                    Dq = 0.1diffusion_parameters / q + 1.1diffusion_parameters / q^2 - 0.2diffusion_parameters / q^3
                    D′q = -0.1diffusion_parameters / q^2 - 2.2diffusion_parameters / q^3 + 0.6diffusion_parameters / q^4
                    _b1[ctr] = ∂q∂t - D′q * ∂q∂x^2 - Dq * ∂²q∂x²
                    _idx_map[ctr] = (k, i, j)
                    ctr += 1
                end
            end
        end
        _A1 = _A1[1:ctr-1, :]
        _b1 = _b1[1:ctr-1]
        _A2 = zeros(sum(length.(esol.u)), 2)
        _b2 = zeros(sum(length.(esol.u)))
        global ctr = 1
        for k in 1:length(esol)
            sol = esol[k]
            for j in 2:length(sol)
                global ctr
                q = EQL.cell_density(sol, length(sol.u[j]), j)
                ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
                _A2[ctr, 1] = q^2
                _A2[ctr, 2] = q^3
                _b2[ctr] = ∂q∂x
                _idx_map[size(_A1, 1)+ctr] = (k, 0, j)
                ctr += 1
            end
        end
        _A2 = _A2[1:ctr-1, :]
        _b2 = _b2[1:ctr-1]
        _A3 = zeros(sum(length.(esol.u)), 3)
        _b3 = zeros(sum(length.(esol.u)))
        global ctr = 1
        for k in 1:length(esol)
            sol = esol[k]
            for j in 2:length(sol)
                global ctr
                q = EQL.cell_density(sol, length(sol.u[j]), j)
                ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
                dLdt = EQL.cell_dLdt(sol, j)
                _A3[ctr, 1] = -1 / q * ∂q∂x
                _A3[ctr, 2] = -1 / q^2 * ∂q∂x
                _A3[ctr, 3] = -1 / q^3 * ∂q∂x
                _b3[ctr] = q * dLdt
                _idx_map[size(_A1, 1)+size(_A2, 1)+ctr] = (k, -1, j)
                ctr += 1
            end
        end
        _A3 = _A3[1:ctr-1, :]
        _b3 = _b3[1:ctr-1]
        _A = (Matrix ∘ BlockDiagonal)([_A1, _A2, _A3])
        _b = vcat(_b1, _b2, _b3)
        @test A ≈ _A
        @test b ≈ _b
        @test idx_map == _idx_map
        @test valid_time_indices == 2:length(esol[1])
    end

    # Fixing reaction parameters
    begin
        A, b, idx_map, valid_time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
            diffusion_basis, diffusion_parameters,
            reaction_basis, reaction_parameters, reaction_theta=[0.5, -0.7, 1.3],
            rhs_basis, rhs_parameters, average=Val(false),
            moving_boundary_basis, moving_boundary_parameters,
            threshold_tol=(q=0.3, dt=0.1))
        min_q, max_q = quantile(all_q, [0.3, 0.7])
        min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [0.1, 0.9])
        @test qmin == minimum(all_q)
        @test qmax == maximum(all_q)
        @test esol === _sol
        _neqs = sum(sum(length.(sol.u)) for sol in esol)
        _A1 = zeros(_neqs, 3)
        _b1 = zeros(_neqs)
        _idx_map = Bijection{Int,NTuple{3,Int}}()
        global ctr = 1
        for k in 1:length(esol)
            for j in 2:length(esol[k])
                for i in 1:length(esol[k][j])
                    global ctr
                    q = EQL.cell_density(esol[k], i, j)
                    ∂q∂x = EQL.cell_∂q∂x(esol[k], i, j)
                    ∂q∂t = EQL.cell_∂q∂t(esol[k], i, j)
                    ∂²q∂x² = EQL.cell_∂²q∂x²(esol[k], i, j)
                    _A1[ctr, 1] = -diffusion_parameters / q^2 * ∂q∂x^2 + diffusion_parameters / q * ∂²q∂x²
                    _A1[ctr, 2] = -2diffusion_parameters / q^3 * ∂q∂x^2 + diffusion_parameters / q^2 * ∂²q∂x²
                    _A1[ctr, 3] = -3diffusion_parameters / q^4 * ∂q∂x^2 + diffusion_parameters / q^3 * ∂²q∂x²
                    _b1[ctr] = ∂q∂t - 0.5Gp.β * q + 0.7Gp.β * q^2 - 1.3Gp.β * q^3
                    if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq) && (0 ≤ esol[k][j][i] ≤ (1 - 0.1) * esol[k][j][end])
                        _idx_map[ctr] = (k, i, j)
                        ctr += 1
                    end
                end
            end
        end
        _A1 = _A1[1:ctr-1, :]
        _b1 = _b1[1:ctr-1]
        _A2 = zeros(sum(length.(esol.u)), 2)
        _b2 = zeros(sum(length.(esol.u)))
        global ctr = 1
        for k in 1:length(esol)
            sol = esol[k]
            for j in 2:length(sol)
                global ctr
                q = EQL.cell_density(sol, length(sol.u[j]), j)
                ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
                _A2[ctr, 1] = q^2
                _A2[ctr, 2] = q^3
                _b2[ctr] = ∂q∂x
                _idx_map[size(_A1, 1)+ctr] = (k, 0, j)
                ctr += 1
            end
        end
        _A2 = _A2[1:ctr-1, :]
        _b2 = _b2[1:ctr-1]
        _A3 = zeros(sum(length.(esol.u)), 3)
        _b3 = zeros(sum(length.(esol.u)))
        global ctr = 1
        for k in 1:length(esol)
            sol = esol[k]
            for j in 2:length(sol)
                global ctr
                q = EQL.cell_density(sol, length(sol.u[j]), j)
                ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
                dLdt = EQL.cell_dLdt(sol, j)
                _A3[ctr, 1] = -1 / q * ∂q∂x
                _A3[ctr, 2] = -1 / q^2 * ∂q∂x
                _A3[ctr, 3] = -1 / q^3 * ∂q∂x
                _b3[ctr] = q * dLdt
                _idx_map[size(_A1, 1)+size(_A2, 1)+ctr] = (k, -1, j)
                ctr += 1
            end
        end
        _A3 = _A3[1:ctr-1, :]
        _b3 = _b3[1:ctr-1]
        _A = (Matrix ∘ BlockDiagonal)([_A1, _A2, _A3])
        _b = vcat(_b1, _b2, _b3)
        @test A ≈ _A
        @test b ≈ _b
        @test idx_map == _idx_map
        @test valid_time_indices == 2:length(esol[1])
    end

    # Giving a time range 
    begin
        all_q = Float64[]
        all_∂ₜq = Float64[]
        all_∂ₓq = Float64[]
        all_∂²ₓq² = Float64[]
        all_∂ₓq_bc = Float64[]
        all_dLdt = Float64[]
        for k in 1:length(esol)
            sol = esol[k]
            for j in 2:length(sol)
                if 14.0 ≤ sol.t[j] ≤ 28.0
                    for i in 1:length(sol[j])
                        push!(all_q, EQL.cell_density(sol, i, j))
                        push!(all_∂ₜq, EQL.cell_∂q∂t(sol, i, j))
                        push!(all_∂ₓq, EQL.cell_∂q∂x(sol, i, j))
                        push!(all_∂²ₓq², EQL.cell_∂²q∂x²(sol, i, j))
                    end
                    push!(all_∂ₓq_bc, EQL.cell_∂q∂x(sol, length(sol[j]), j))
                    push!(all_dLdt, EQL.cell_dLdt(sol, j))
                end
            end
        end
        A, b, idx_map, time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
            diffusion_basis, diffusion_parameters,
            reaction_basis, reaction_parameters, time_range=(14.0, 28.0),
            rhs_basis, rhs_parameters, average=Val(false),
            moving_boundary_basis, moving_boundary_parameters,
            threshold_tol=(q=0.3, dt=0.1, dL=0.45))
        min_q, max_q = quantile(all_q, [0.3, 0.7])
        min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [0.1, 0.9])
        min_dLdt, max_dLdt = quantile(abs.(all_dLdt), [0.45, 0.55])
        @test esol === _sol
        _neqs = sum(sum(length.(sol.u)) for sol in esol)
        _A1 = zeros(_neqs, 6)
        _b1 = zeros(_neqs)
        _idx_map = Bijection{Int,NTuple{3,Int}}()
        global ctr = 1
        for k in 1:length(esol)
            for j in 2:length(esol[k])
                if 14.0 ≤ esol[k].t[j] ≤ 28.0
                    for i in 1:length(esol[k][j])
                        global ctr
                        q = EQL.cell_density(esol[k], i, j)
                        ∂q∂x = EQL.cell_∂q∂x(esol[k], i, j)
                        ∂q∂t = EQL.cell_∂q∂t(esol[k], i, j)
                        ∂²q∂x² = EQL.cell_∂²q∂x²(esol[k], i, j)
                        _A1[ctr, 1] = -diffusion_parameters / q^2 * ∂q∂x^2 + diffusion_parameters / q * ∂²q∂x²
                        _A1[ctr, 2] = -2diffusion_parameters / q^3 * ∂q∂x^2 + diffusion_parameters / q^2 * ∂²q∂x²
                        _A1[ctr, 3] = -3diffusion_parameters / q^4 * ∂q∂x^2 + diffusion_parameters / q^3 * ∂²q∂x²
                        _A1[ctr, 4] = Gp.β * q
                        _A1[ctr, 5] = Gp.β * q^2
                        _A1[ctr, 6] = Gp.β * q^3
                        _b1[ctr] = ∂q∂t
                        if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq)
                            _idx_map[ctr] = (k, i, j)
                            ctr += 1
                        end
                    end
                end
            end
        end
        _A1 = _A1[1:ctr-1, :]
        _b1 = _b1[1:ctr-1]
        _A2 = zeros(sum(length.(esol.u)), 2)
        _b2 = zeros(sum(length.(esol.u)))
        global ctr = 1
        for k in 1:length(esol)
            sol = esol[k]
            for j in 2:length(sol)
                if 14.0 ≤ sol.t[j] ≤ 28.0
                    global ctr
                    q = EQL.cell_density(sol, length(sol.u[j]), j)
                    ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
                    dLdt = EQL.cell_dLdt(sol, j)
                    _A2[ctr, 1] = q^2
                    _A2[ctr, 2] = q^3
                    _b2[ctr] = ∂q∂x
                    _idx_map[size(_A1, 1)+ctr] = (k, 0, j)
                    ctr += 1
                end
            end
        end
        _A2 = _A2[1:ctr-1, :]
        _b2 = _b2[1:ctr-1]
        _A3 = zeros(sum(length.(esol.u)), 3)
        _b3 = zeros(sum(length.(esol.u)))
        global ctr = 1
        for k in 1:length(esol)
            sol = esol[k]
            for j in 2:length(sol)
                if 14.0 ≤ sol.t[j] ≤ 28.0
                    global ctr
                    q = EQL.cell_density(sol, length(sol.u[j]), j)
                    ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
                    dLdt = EQL.cell_dLdt(sol, j)
                    _A3[ctr, 1] = -1 / q * ∂q∂x
                    _A3[ctr, 2] = -1 / q^2 * ∂q∂x
                    _A3[ctr, 3] = -1 / q^3 * ∂q∂x
                    _b3[ctr] = q * dLdt
                    if (min_dLdt ≤ abs(dLdt) ≤ max_dLdt)
                        _idx_map[size(_A1, 1)+size(_A2, 1)+ctr] = (k, -1, j)
                        ctr += 1
                    end
                end
            end
        end
        _A3 = _A3[1:ctr-1, :]
        _b3 = _b3[1:ctr-1]
        _A = (Matrix ∘ BlockDiagonal)([_A1, _A2, _A3])
        _b = vcat(_b1, _b2, _b3)
        @test A ≈ _A
        @test b ≈ _b
        @test idx_map == _idx_map
        @test all(j -> 14.0 ≤ esol[1].t[j] ≤ 28.0, time_indices)
        @test all(j -> (esol[1].t[j] < 14.0 || esol[1].t[j] > 28.0), setdiff(1:length(esol[1]), time_indices))
    end

    # Averaging 
    begin
        indices = rand(eachindex(esol), 20)
        sol = EQL.AveragedODESolution(esol, 200, indices)
        all_q = Float64[]
        all_∂ₜq = Float64[]
        all_∂ₓq = Float64[]
        all_∂²ₓq² = Float64[]
        all_∂ₓq_bc = Float64[]
        all_dLdt = Float64[]
        for j in 2:length(sol)
            for i in 1:length(sol.u[j])
                push!(all_q, EQL.cell_density(sol, i, j))
                push!(all_∂ₜq, EQL.cell_∂q∂t(sol, i, j))
                push!(all_∂ₓq, EQL.cell_∂q∂x(sol, i, j))
                push!(all_∂²ₓq², EQL.cell_∂²q∂x²(sol, i, j))
            end
            push!(all_∂ₓq_bc, EQL.cell_∂q∂x(sol, length(sol.u[j]), j))
            push!(all_dLdt, EQL.cell_dLdt(sol, j))
        end
        A, b, idx_map, time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
            diffusion_basis, diffusion_parameters,
            reaction_basis, reaction_parameters,
            rhs_basis, rhs_parameters, num_knots=200, simulation_indices=indices,
            moving_boundary_basis, moving_boundary_parameters,
            threshold_tol=(q=0.3, dt=0.1, dx=0.2, x=0.2))
        min_q, max_q = quantile(all_q, [0.3, 0.7])
        min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [0.1, 0.9])
        min_∂ₓq, max_∂ₓq = quantile(abs.(all_∂ₓq), [0.2, 0.8])
        @test sol.q == _sol.q
        @test sol.u == _sol.u
        @test sol.t == _sol.t
        @test sol.cell_sol === _sol.cell_sol
        _neqs = sum(length.(sol.u))
        _A1 = zeros(_neqs, 6)
        _b1 = zeros(_neqs)
        _idx_map = Bijection{Int,NTuple{3,Int}}()
        global ctr = 1
        for j in 2:length(sol)
            for i in 1:length(sol.u[j])
                global ctr
                q = EQL.cell_density(sol, i, j)
                ∂q∂x = EQL.cell_∂q∂x(sol, i, j)
                ∂q∂t = EQL.cell_∂q∂t(sol, i, j)
                ∂²q∂x² = EQL.cell_∂²q∂x²(sol, i, j)
                _A1[ctr, 1] = -diffusion_parameters / q^2 * ∂q∂x^2 + diffusion_parameters / q * ∂²q∂x²
                _A1[ctr, 2] = -2diffusion_parameters / q^3 * ∂q∂x^2 + diffusion_parameters / q^2 * ∂²q∂x²
                _A1[ctr, 3] = -3diffusion_parameters / q^4 * ∂q∂x^2 + diffusion_parameters / q^3 * ∂²q∂x²
                _A1[ctr, 4] = Gp.β * q
                _A1[ctr, 5] = Gp.β * q^2
                _A1[ctr, 6] = Gp.β * q^3
                _b1[ctr] = ∂q∂t
                if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq) && (min_∂ₓq ≤ abs(∂q∂x) ≤ max_∂ₓq) && (0 ≤ sol.u[j][i] ≤ (1 - 0.2) * sol.u[j][end])
                    _idx_map[ctr] = (1, i, j)
                    ctr += 1
                end
            end
        end
        _A1 = _A1[1:ctr-1, :]
        _b1 = _b1[1:ctr-1]
        _A2 = zeros(length(sol), 2)
        _b2 = zeros(length(sol))
        global ctr = 1
        for j in 2:length(sol)
            global ctr
            q = EQL.cell_density(sol, length(sol.u[j]), j)
            ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
            _A2[ctr, 1] = q^2
            _A2[ctr, 2] = q^3
            _b2[ctr] = ∂q∂x
            _idx_map[size(_A1, 1)+ctr] = (1, 0, j)
            ctr += 1
        end
        _A2 = _A2[1:ctr-1, :]
        _b2 = _b2[1:ctr-1]
        _A3 = zeros(sum(length.(esol.u)), 3)
        _b3 = zeros(sum(length.(esol.u)))
        global ctr = 1
        for j in 2:length(sol)
            global ctr
            q = EQL.cell_density(sol, length(sol.u[j]), j)
            ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
            dLdt = EQL.cell_dLdt(sol, j)
            _A3[ctr, 1] = -1 / q * ∂q∂x
            _A3[ctr, 2] = -1 / q^2 * ∂q∂x
            _A3[ctr, 3] = -1 / q^3 * ∂q∂x
            _b3[ctr] = q * dLdt
            _idx_map[size(_A1, 1)+size(_A2, 1)+ctr] = (1, -1, j)
            ctr += 1
        end
        _A3 = _A3[1:ctr-1, :]
        _b3 = _b3[1:ctr-1]
        _A = (Matrix ∘ BlockDiagonal)([_A1, _A2, _A3])
        _b = vcat(_b1, _b2, _b3)
        @test A ≈ _A
        @test b ≈ _b
        @test idx_map == _idx_map
        @test time_indices == 2:length(esol[1])
    end

    # Test the averaged defaults
    begin
        indices = eachindex(esol)
        num_knots = 100
        avg_interp_fnc = LinearInterpolation{true}
        sol = EQL.AveragedODESolution(esol, num_knots, indices, avg_interp_fnc, mean)
        A, b, idx_map, time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
            diffusion_basis, diffusion_parameters,
            reaction_basis, reaction_parameters,
            rhs_basis, rhs_parameters,
            moving_boundary_basis, moving_boundary_parameters)
        @test sol.q == _sol.q
        @test sol.u == _sol.u
        @test sol.t == _sol.t
        @test sol.cell_sol === _sol.cell_sol
        _neqs = sum(length.(sol.u))
        _A1 = zeros(_neqs, 6)
        _b1 = zeros(_neqs)
        _idx_map = Bijection{Int,NTuple{3,Int}}()
        global ctr = 1
        for j in 2:length(sol)
            for i in 1:length(sol.u[j])
                global ctr
                q = EQL.cell_density(sol, i, j)
                ∂q∂x = EQL.cell_∂q∂x(sol, i, j)
                ∂q∂t = EQL.cell_∂q∂t(sol, i, j)
                ∂²q∂x² = EQL.cell_∂²q∂x²(sol, i, j)
                _A1[ctr, 1] = -diffusion_parameters / q^2 * ∂q∂x^2 + diffusion_parameters / q * ∂²q∂x²
                _A1[ctr, 2] = -2diffusion_parameters / q^3 * ∂q∂x^2 + diffusion_parameters / q^2 * ∂²q∂x²
                _A1[ctr, 3] = -3diffusion_parameters / q^4 * ∂q∂x^2 + diffusion_parameters / q^3 * ∂²q∂x²
                _A1[ctr, 4] = Gp.β * q
                _A1[ctr, 5] = Gp.β * q^2
                _A1[ctr, 6] = Gp.β * q^3
                _b1[ctr] = ∂q∂t
                _idx_map[ctr] = (1, i, j)
                ctr += 1
            end
        end
        _A1 = _A1[1:ctr-1, :]
        _b1 = _b1[1:ctr-1]
        _A2 = zeros(length(sol), 2)
        _b2 = zeros(length(sol))
        global ctr = 1
        for j in 2:length(sol)
            global ctr
            q = EQL.cell_density(sol, length(sol.u[j]), j)
            ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
            _A2[ctr, 1] = q^2
            _A2[ctr, 2] = q^3
            _b2[ctr] = ∂q∂x
            _idx_map[size(_A1, 1)+ctr] = (1, 0, j)
            ctr += 1
        end
        _A2 = _A2[1:ctr-1, :]
        _b2 = _b2[1:ctr-1]
        _A3 = zeros(sum(length.(esol.u)), 3)
        _b3 = zeros(sum(length.(esol.u)))
        global ctr = 1
        for j in 2:length(sol)
            global ctr
            q = EQL.cell_density(sol, length(sol.u[j]), j)
            ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
            dLdt = EQL.cell_dLdt(sol, j)
            _A3[ctr, 1] = -1 / q * ∂q∂x
            _A3[ctr, 2] = -1 / q^2 * ∂q∂x
            _A3[ctr, 3] = -1 / q^3 * ∂q∂x
            _b3[ctr] = q * dLdt
            _idx_map[size(_A1, 1)+size(_A2, 1)+ctr] = (1, -1, j)
            ctr += 1
        end
        _A3 = _A3[1:ctr-1, :]
        _b3 = _b3[1:ctr-1]
        _A = (Matrix ∘ BlockDiagonal)([_A1, _A2, _A3])
        _b = vcat(_b1, _b2, _b3)
        @test A ≈ _A
        @test b ≈ _b
        @test idx_map == _idx_map
        @test time_indices == 2:length(esol[1])
    end

    # Providing a subset of simulation indices for aggregating  
    begin
        simulation_indices = rand(eachindex(esol), 20) |> unique!
        all_q = Float64[]
        all_∂ₜq = Float64[]
        all_∂ₓq = Float64[]
        all_∂²ₓq² = Float64[]
        all_∂ₓq_bc = Float64[]
        all_dLdt = Float64[]
        for k in simulation_indices
            sol = esol[k]
            for j in 2:length(sol)
                if 14.0 ≤ sol.t[j] ≤ 28.0
                    for i in 1:length(sol[j])
                        push!(all_q, EQL.cell_density(sol, i, j))
                        push!(all_∂ₜq, EQL.cell_∂q∂t(sol, i, j))
                        push!(all_∂ₓq, EQL.cell_∂q∂x(sol, i, j))
                        push!(all_∂²ₓq², EQL.cell_∂²q∂x²(sol, i, j))
                    end
                    push!(all_∂ₓq_bc, EQL.cell_∂q∂x(sol, length(sol[j]), j))
                    push!(all_dLdt, EQL.cell_dLdt(sol, j))
                end
            end
        end
        A, b, idx_map, time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
            diffusion_basis, diffusion_parameters,
            reaction_basis, reaction_parameters, time_range=(14.0, 28.0),
            rhs_basis, rhs_parameters, simulation_indices, average=Val(false),
            moving_boundary_basis, moving_boundary_parameters,
            threshold_tol=(q=0.3, dt=0.1, dx_bc=0.45, x=0.6))
        min_q, max_q = quantile(all_q, [0.3, 0.7])
        min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [0.1, 0.9])
        min_∂ₓq_bc, max_∂ₓq_bc = quantile(abs.(all_∂ₓq_bc), [0.45, 0.55])
        @test esol === _sol
        _neqs = sum(sum(length.(sol.u)) for sol in esol[simulation_indices])
        __neqs = sum(sum(length.(sol.u) for sol in esol))
        _A1 = zeros(_neqs, 6)
        _b1 = zeros(_neqs)
        _idx_map = Bijection{Int,NTuple{3,Int}}()
        global ctr = 1
        for k in simulation_indices
            for j in 2:length(esol[k])
                if 14.0 ≤ esol[k].t[j] ≤ 28.0
                    for i in 1:length(esol[k][j])
                        global ctr
                        q = EQL.cell_density(esol[k], i, j)
                        ∂q∂x = EQL.cell_∂q∂x(esol[k], i, j)
                        ∂q∂t = EQL.cell_∂q∂t(esol[k], i, j)
                        ∂²q∂x² = EQL.cell_∂²q∂x²(esol[k], i, j)
                        _A1[ctr, 1] = -diffusion_parameters / q^2 * ∂q∂x^2 + diffusion_parameters / q * ∂²q∂x²
                        _A1[ctr, 2] = -2diffusion_parameters / q^3 * ∂q∂x^2 + diffusion_parameters / q^2 * ∂²q∂x²
                        _A1[ctr, 3] = -3diffusion_parameters / q^4 * ∂q∂x^2 + diffusion_parameters / q^3 * ∂²q∂x²
                        _A1[ctr, 4] = Gp.β * q
                        _A1[ctr, 5] = Gp.β * q^2
                        _A1[ctr, 6] = Gp.β * q^3
                        _b1[ctr] = ∂q∂t
                        if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq) && (0 ≤ esol[k][j][i] ≤ 0.4esol[k][j][end])
                            _idx_map[ctr] = (k, i, j)
                            ctr += 1
                        end
                    end
                end
            end
        end
        _A1 = _A1[1:ctr-1, :]
        _b1 = _b1[1:ctr-1]
        _A2 = zeros(sum(length.(esol.u)), 2)
        _b2 = zeros(sum(length.(esol.u)))
        global ctr = 1
        for k in simulation_indices
            sol = esol[k]
            for j in 2:length(sol)
                if 14.0 ≤ sol.t[j] ≤ 28.0
                    global ctr
                    q = EQL.cell_density(sol, length(sol.u[j]), j)
                    ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
                    _A2[ctr, 1] = q^2
                    _A2[ctr, 2] = q^3
                    _b2[ctr] = ∂q∂x
                    if (min_∂ₓq_bc ≤ abs(∂q∂x) ≤ max_∂ₓq_bc)
                        _idx_map[size(_A1, 1)+ctr] = (k, 0, j)
                        ctr += 1
                    end
                end
            end
        end
        _A2 = _A2[1:ctr-1, :]
        _b2 = _b2[1:ctr-1]
        _A3 = zeros(sum(length.(esol.u)), 3)
        _b3 = zeros(sum(length.(esol.u)))
        global ctr = 1
        for k in simulation_indices
            sol = esol[k]
            for j in 2:length(sol)
                if 14.0 ≤ sol.t[j] ≤ 28.0
                    global ctr
                    q = EQL.cell_density(sol, length(sol.u[j]), j)
                    ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
                    dLdt = EQL.cell_dLdt(sol, j)
                    _A3[ctr, 1] = -1 / q * ∂q∂x
                    _A3[ctr, 2] = -1 / q^2 * ∂q∂x
                    _A3[ctr, 3] = -1 / q^3 * ∂q∂x
                    _b3[ctr] = q * dLdt
                    _idx_map[size(_A1, 1)+size(_A2, 1)+ctr] = (k, -1, j)
                    ctr += 1
                end
            end
        end
        _A3 = _A3[1:ctr-1, :]
        _b3 = _b3[1:ctr-1]
        _A = (Matrix ∘ BlockDiagonal)([_A1, _A2, _A3])
        _b = vcat(_b1, _b2, _b3)
        @test A ≈ _A
        @test b ≈ _b
        @test idx_map == _idx_map
        @test all(j -> 14.0 ≤ esol[1].t[j] ≤ 28.0, time_indices)
        @test all(j -> (esol[1].t[j] < 14.0 || esol[1].t[j] > 28.0), setdiff(1:length(esol[1]), time_indices))
    end

    # Averaging with a CubicSpline
    begin
        indices = rand(eachindex(esol), 20)
        sol = EQL.AveragedODESolution(esol, 200, indices, CubicSpline)
        all_q = Float64[]
        all_∂ₜq = Float64[]
        all_∂ₓq = Float64[]
        all_∂²ₓq² = Float64[]
        all_∂ₓq_bc = Float64[]
        all_dLdt = Float64[]
        for j in 2:length(sol)
            for i in 1:length(sol.u[j])
                push!(all_q, EQL.cell_density(sol, i, j))
                push!(all_∂ₜq, EQL.cell_∂q∂t(sol, i, j))
                push!(all_∂ₓq, EQL.cell_∂q∂x(sol, i, j))
                push!(all_∂²ₓq², EQL.cell_∂²q∂x²(sol, i, j))
            end
            push!(all_∂ₓq_bc, EQL.cell_∂q∂x(sol, length(sol.u[j]), j))
            push!(all_dLdt, EQL.cell_dLdt(sol, j))
        end
        A, b, idx_map, time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
            diffusion_basis, diffusion_parameters,
            reaction_basis, reaction_parameters, avg_interp_fnc=CubicSpline,
            rhs_basis, rhs_parameters, num_knots=200, simulation_indices=indices,
            moving_boundary_basis, moving_boundary_parameters,
            threshold_tol=(q=0.3, dt=0.1, dx=0.2))
        min_q, max_q = quantile(all_q, [0.3, 0.7])
        min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [0.1, 0.9])
        min_∂ₓq, max_∂ₓq = quantile(abs.(all_∂ₓq), [0.2, 0.8])
        @test sol.q == _sol.q
        @test sol.u == _sol.u
        @test sol.t == _sol.t
        @test sol.cell_sol === _sol.cell_sol
        _neqs = sum(length.(sol.u))
        _A1 = zeros(_neqs, 6)
        _b1 = zeros(_neqs)
        _idx_map = Bijection{Int,NTuple{3,Int}}()
        global ctr = 1
        for j in 2:length(sol)
            for i in 1:length(sol.u[j])
                global ctr
                q = EQL.cell_density(sol, i, j)
                ∂q∂x = EQL.cell_∂q∂x(sol, i, j)
                ∂q∂t = EQL.cell_∂q∂t(sol, i, j)
                ∂²q∂x² = EQL.cell_∂²q∂x²(sol, i, j)
                _A1[ctr, 1] = -diffusion_parameters / q^2 * ∂q∂x^2 + diffusion_parameters / q * ∂²q∂x²
                _A1[ctr, 2] = -2diffusion_parameters / q^3 * ∂q∂x^2 + diffusion_parameters / q^2 * ∂²q∂x²
                _A1[ctr, 3] = -3diffusion_parameters / q^4 * ∂q∂x^2 + diffusion_parameters / q^3 * ∂²q∂x²
                _A1[ctr, 4] = Gp.β * q
                _A1[ctr, 5] = Gp.β * q^2
                _A1[ctr, 6] = Gp.β * q^3
                _b1[ctr] = ∂q∂t
                if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq) && (min_∂ₓq ≤ abs(∂q∂x) ≤ max_∂ₓq)
                    _idx_map[ctr] = (1, i, j)
                    ctr += 1
                end
            end
        end
        _A1 = _A1[1:ctr-1, :]
        _b1 = _b1[1:ctr-1]
        _A2 = zeros(length(sol), 2)
        _b2 = zeros(length(sol))
        global ctr = 1
        for j in 2:length(sol)
            global ctr
            q = EQL.cell_density(sol, length(sol.u[j]), j)
            ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
            _A2[ctr, 1] = q^2
            _A2[ctr, 2] = q^3
            _b2[ctr] = ∂q∂x
            _idx_map[size(_A1, 1)+ctr] = (1, 0, j)
            ctr += 1
        end
        _A2 = _A2[1:ctr-1, :]
        _b2 = _b2[1:ctr-1]
        _A3 = zeros(sum(length.(esol.u)), 3)
        _b3 = zeros(sum(length.(esol.u)))
        global ctr = 1
        for j in 2:length(sol)
            global ctr
            q = EQL.cell_density(sol, length(sol.u[j]), j)
            ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
            dLdt = EQL.cell_dLdt(sol, j)
            _A3[ctr, 1] = -1 / q * ∂q∂x
            _A3[ctr, 2] = -1 / q^2 * ∂q∂x
            _A3[ctr, 3] = -1 / q^3 * ∂q∂x
            _b3[ctr] = q * dLdt
            _idx_map[size(_A1, 1)+size(_A2, 1)+ctr] = (1, -1, j)
            ctr += 1
        end
        _A3 = _A3[1:ctr-1, :]
        _b3 = _b3[1:ctr-1]
        _A = (Matrix ∘ BlockDiagonal)([_A1, _A2, _A3])
        _b = vcat(_b1, _b2, _b3)
        @test A ≈ _A
        @test b ≈ _b
        @test idx_map == _idx_map
        @test time_indices == 2:length(esol[1])
    end

    # Averaging with stat = (minimum, maximum)
    begin
        indices = rand(eachindex(esol), 20)
        sol = EQL.AveragedODESolution(esol, 200, indices, LinearInterpolation, (minimum, maximum))
        all_q = Float64[]
        all_∂ₜq = Float64[]
        all_∂ₓq = Float64[]
        all_∂²ₓq² = Float64[]
        all_∂ₓq_bc = Float64[]
        all_dLdt = Float64[]
        for j in 2:length(sol)
            for i in 1:length(sol.u[j])
                push!(all_q, EQL.cell_density(sol, i, j))
                push!(all_∂ₜq, EQL.cell_∂q∂t(sol, i, j))
                push!(all_∂ₓq, EQL.cell_∂q∂x(sol, i, j))
                push!(all_∂²ₓq², EQL.cell_∂²q∂x²(sol, i, j))
            end
            push!(all_∂ₓq_bc, EQL.cell_∂q∂x(sol, length(sol.u[j]), j))
            push!(all_dLdt, EQL.cell_dLdt(sol, j))
        end
        A, b, idx_map, time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
            diffusion_basis, diffusion_parameters, stat=(minimum, maximum),
            reaction_basis, reaction_parameters,
            rhs_basis, rhs_parameters, num_knots=200, simulation_indices=indices,
            moving_boundary_basis, moving_boundary_parameters,
            threshold_tol=(q=0.3, dt=0.1, dx=0.2, dL=0.1))
        min_q, max_q = quantile(all_q, [0.3, 0.7])
        min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [0.1, 0.9])
        min_∂ₓq, max_∂ₓq = quantile(abs.(all_∂ₓq), [0.2, 0.8])
        min_dLdt, max_dLdt = quantile(abs.(all_dLdt), [0.1, 0.9])
        @test sol.q == _sol.q
        @test sol.u == _sol.u
        @test sol.t == _sol.t
        @test sol.cell_sol === _sol.cell_sol
        _neqs = sum(length.(sol.u))
        _A1 = zeros(_neqs, 6)
        _b1 = zeros(_neqs)
        _idx_map = Bijection{Int,NTuple{3,Int}}()
        global ctr = 1
        for j in 2:length(sol)
            for i in 1:length(sol.u[j])
                global ctr
                q = EQL.cell_density(sol, i, j)
                ∂q∂x = EQL.cell_∂q∂x(sol, i, j)
                ∂q∂t = EQL.cell_∂q∂t(sol, i, j)
                ∂²q∂x² = EQL.cell_∂²q∂x²(sol, i, j)
                _A1[ctr, 1] = -diffusion_parameters / q^2 * ∂q∂x^2 + diffusion_parameters / q * ∂²q∂x²
                _A1[ctr, 2] = -2diffusion_parameters / q^3 * ∂q∂x^2 + diffusion_parameters / q^2 * ∂²q∂x²
                _A1[ctr, 3] = -3diffusion_parameters / q^4 * ∂q∂x^2 + diffusion_parameters / q^3 * ∂²q∂x²
                _A1[ctr, 4] = Gp.β * q
                _A1[ctr, 5] = Gp.β * q^2
                _A1[ctr, 6] = Gp.β * q^3
                _b1[ctr] = ∂q∂t
                if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq) && (min_∂ₓq ≤ abs(∂q∂x) ≤ max_∂ₓq)
                    _idx_map[ctr] = (1, i, j)
                    ctr += 1
                end
            end
        end
        _A1 = _A1[1:ctr-1, :]
        _b1 = _b1[1:ctr-1]
        _A2 = zeros(length(sol), 2)
        _b2 = zeros(length(sol))
        global ctr = 1
        for j in 2:length(sol)
            global ctr
            q = EQL.cell_density(sol, length(sol.u[j]), j)
            ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
            dLdt = EQL.cell_dLdt(sol, j)
            _A2[ctr, 1] = q^2
            _A2[ctr, 2] = q^3
            _b2[ctr] = ∂q∂x
            _idx_map[size(_A1, 1)+ctr] = (1, 0, j)
            ctr += 1
        end
        _A2 = _A2[1:ctr-1, :]
        _b2 = _b2[1:ctr-1]
        _A3 = zeros(sum(length.(esol.u)), 3)
        _b3 = zeros(sum(length.(esol.u)))
        global ctr = 1
        for j in 2:length(sol)
            global ctr
            q = EQL.cell_density(sol, length(sol.u[j]), j)
            ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
            dLdt = EQL.cell_dLdt(sol, j)
            _A3[ctr, 1] = -1 / q * ∂q∂x
            _A3[ctr, 2] = -1 / q^2 * ∂q∂x
            _A3[ctr, 3] = -1 / q^3 * ∂q∂x
            _b3[ctr] = q * dLdt
            if (min_dLdt ≤ abs(dLdt) ≤ max_dLdt)
                _idx_map[size(_A1, 1)+size(_A2, 1)+ctr] = (1, -1, j)
                ctr += 1
            end
        end
        _A3 = _A3[1:ctr-1, :]
        _b3 = _b3[1:ctr-1]
        _A = (Matrix ∘ BlockDiagonal)([_A1, _A2, _A3])
        _b = vcat(_b1, _b2, _b3)
        @test A ≈ _A
        @test b ≈ _b
        @test idx_map == _idx_map
        @test time_indices == 2:length(esol[1])
    end

    # Fixing RHS parameters 
    begin
        indices = rand(eachindex(esol), 20)
        sol = EQL.AveragedODESolution(esol, 250, indices)
        all_q = Float64[]
        all_∂ₜq = Float64[]
        all_∂ₓq = Float64[]
        all_∂²ₓq² = Float64[]
        all_∂ₓq_bc = Float64[]
        all_dLdt = Float64[]
        for j in 2:length(sol)
            if 0.5 ≤ sol.t[j] ≤ 15.0
                for i in 1:length(sol.u[j])
                    push!(all_q, EQL.cell_density(sol, i, j))
                    push!(all_∂ₜq, EQL.cell_∂q∂t(sol, i, j))
                    push!(all_∂ₓq, EQL.cell_∂q∂x(sol, i, j))
                    push!(all_∂²ₓq², EQL.cell_∂²q∂x²(sol, i, j))
                end
                push!(all_∂ₓq_bc, EQL.cell_∂q∂x(sol, length(sol.u[j]), j))
                push!(all_dLdt, EQL.cell_dLdt(sol, j))
            end
        end
        A, b, idx_map, time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
            diffusion_basis, diffusion_parameters,
            reaction_basis, reaction_parameters, rhs_theta=[2.0, -2resting_spring_length],
            rhs_basis, rhs_parameters, num_knots=250, simulation_indices=indices,
            moving_boundary_basis, moving_boundary_parameters, time_range=(0.5, 15.0),
            threshold_tol=(q=0.3, dt=0.1, dx=0.2, dL=0.1, dx2=0.3))
        min_q, max_q = quantile(all_q, [0.3, 0.7])
        min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [0.1, 0.9])
        min_∂ₓq, max_∂ₓq = quantile(abs.(all_∂ₓq), [0.2, 0.8])
        min_dLdt, max_dLdt = quantile(abs.(all_dLdt), [0.1, 0.9])
        min_∂²ₓq², max_∂²ₓq² = quantile(abs.(all_∂²ₓq²), [0.3, 0.7])
        @test sol.q == _sol.q
        @test sol.u == _sol.u
        @test sol.t == _sol.t
        @test sol.cell_sol === _sol.cell_sol
        _neqs = sum(length.(sol.u))
        _A1 = zeros(_neqs, 6)
        _b1 = zeros(_neqs)
        _idx_map = Bijection{Int,NTuple{3,Int}}()
        global ctr = 1
        for j in 2:length(sol)
            if 0.5 ≤ sol.t[j] ≤ 15.0
                for i in 1:length(sol.u[j])
                    global ctr
                    q = EQL.cell_density(sol, i, j)
                    ∂q∂x = EQL.cell_∂q∂x(sol, i, j)
                    ∂q∂t = EQL.cell_∂q∂t(sol, i, j)
                    ∂²q∂x² = EQL.cell_∂²q∂x²(sol, i, j)
                    _A1[ctr, 1] = -diffusion_parameters / q^2 * ∂q∂x^2 + diffusion_parameters / q * ∂²q∂x²
                    _A1[ctr, 2] = -2diffusion_parameters / q^3 * ∂q∂x^2 + diffusion_parameters / q^2 * ∂²q∂x²
                    _A1[ctr, 3] = -3diffusion_parameters / q^4 * ∂q∂x^2 + diffusion_parameters / q^3 * ∂²q∂x²
                    _A1[ctr, 4] = Gp.β * q
                    _A1[ctr, 5] = Gp.β * q^2
                    _A1[ctr, 6] = Gp.β * q^3
                    _b1[ctr] = ∂q∂t
                    if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq) && (min_∂ₓq ≤ abs(∂q∂x) ≤ max_∂ₓq) && (min_∂²ₓq² ≤ abs(∂²q∂x²) ≤ max_∂²ₓq²)
                        _idx_map[ctr] = (1, i, j)
                        ctr += 1
                    end
                end
            end
        end
        _A1 = _A1[1:ctr-1, :]
        _b1 = _b1[1:ctr-1]
        _A3 = zeros(sum(length.(esol.u)), 3)
        _b3 = zeros(sum(length.(esol.u)))
        global ctr = 1
        for j in 2:length(sol)
            if 0.5 ≤ sol.t[j] ≤ 15.0
                global ctr
                q = EQL.cell_density(sol, length(sol.u[j]), j)
                ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
                dLdt = EQL.cell_dLdt(sol, j)
                _A3[ctr, 1] = -1 / q * ∂q∂x
                _A3[ctr, 2] = -1 / q^2 * ∂q∂x
                _A3[ctr, 3] = -1 / q^3 * ∂q∂x
                _b3[ctr] = q * dLdt
                if (min_dLdt ≤ abs(dLdt) ≤ max_dLdt)
                    _idx_map[size(_A1, 1)+ctr] = (1, -1, j)
                    ctr += 1
                end
            end
        end
        _A3 = _A3[1:ctr-1, :]
        _b3 = _b3[1:ctr-1]
        _A = (Matrix ∘ BlockDiagonal)([_A1, _A3])
        _b = vcat(_b1, _b3)
        @test A ≈ _A
        @test b ≈ _b
        @test idx_map == _idx_map
        @test all(j -> 0.5 ≤ sol.t[j] ≤ 15.0, time_indices)
        @test all(j -> (sol.t[j] < 0.5 || sol.t[j] > 15.0), setdiff(1:length(sol), time_indices))
    end

    # Fixing moving boundary parameters
    begin
        simulation_indices = rand(eachindex(esol), 20) |> unique!
        all_q = Float64[]
        all_∂ₜq = Float64[]
        all_∂ₓq = Float64[]
        all_∂²ₓq² = Float64[]
        all_∂ₓq_bc = Float64[]
        all_dLdt = Float64[]
        for k in simulation_indices
            sol = esol[k]
            for j in 2:length(sol)
                if 14.0 ≤ sol.t[j] ≤ 28.0
                    for i in 1:length(sol[j])
                        push!(all_q, EQL.cell_density(sol, i, j))
                        push!(all_∂ₜq, EQL.cell_∂q∂t(sol, i, j))
                        push!(all_∂ₓq, EQL.cell_∂q∂x(sol, i, j))
                        push!(all_∂²ₓq², EQL.cell_∂²q∂x²(sol, i, j))
                    end
                    push!(all_∂ₓq_bc, EQL.cell_∂q∂x(sol, length(sol[j]), j))
                    push!(all_dLdt, EQL.cell_dLdt(sol, j))
                end
            end
        end
        A, b, idx_map, time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
            diffusion_basis, diffusion_parameters,
            reaction_basis, reaction_parameters, time_range=(14.0, 28.0),
            rhs_basis, rhs_parameters, simulation_indices, average=Val(false),
            moving_boundary_basis, moving_boundary_parameters, moving_boundary_theta=[0.5, spring_constant, 1.2],
            threshold_tol=(q=0.3, dt=0.1, dx_bc=0.45))
        min_q, max_q = quantile(all_q, [0.3, 0.7])
        min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [0.1, 0.9])
        min_∂ₓq_bc, max_∂ₓq_bc = quantile(abs.(all_∂ₓq_bc), [0.45, 0.55])
        @test esol === _sol
        _neqs = sum(sum(length.(sol.u)) for sol in esol[simulation_indices])
        __neqs = sum(sum(length.(sol.u)) for sol in esol)
        _A1 = zeros(_neqs, 6)
        _b1 = zeros(_neqs)
        _idx_map = Bijection{Int,NTuple{3,Int}}()
        global ctr = 1
        for k in simulation_indices
            for j in 2:length(esol[k])
                if 14.0 ≤ esol[k].t[j] ≤ 28.0
                    for i in 1:length(esol[k][j])
                        global ctr
                        q = EQL.cell_density(esol[k], i, j)
                        ∂q∂x = EQL.cell_∂q∂x(esol[k], i, j)
                        ∂q∂t = EQL.cell_∂q∂t(esol[k], i, j)
                        ∂²q∂x² = EQL.cell_∂²q∂x²(esol[k], i, j)
                        _A1[ctr, 1] = -diffusion_parameters / q^2 * ∂q∂x^2 + diffusion_parameters / q * ∂²q∂x²
                        _A1[ctr, 2] = -2diffusion_parameters / q^3 * ∂q∂x^2 + diffusion_parameters / q^2 * ∂²q∂x²
                        _A1[ctr, 3] = -3diffusion_parameters / q^4 * ∂q∂x^2 + diffusion_parameters / q^3 * ∂²q∂x²
                        _A1[ctr, 4] = Gp.β * q
                        _A1[ctr, 5] = Gp.β * q^2
                        _A1[ctr, 6] = Gp.β * q^3
                        _b1[ctr] = ∂q∂t
                        if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq)
                            _idx_map[ctr] = (k, i, j)
                            ctr += 1
                        end
                    end
                end
            end
        end
        _A1 = _A1[1:ctr-1, :]
        _b1 = _b1[1:ctr-1]
        _A2 = zeros(sum(length.(esol.u)), 2)
        _b2 = zeros(sum(length.(esol.u)))
        global ctr = 1
        for k in simulation_indices
            sol = esol[k]
            for j in 2:length(sol)
                if 14.0 ≤ sol.t[j] ≤ 28.0
                    global ctr
                    q = EQL.cell_density(sol, length(sol.u[j]), j)
                    ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
                    _A2[ctr, 1] = q^2
                    _A2[ctr, 2] = q^3
                    _b2[ctr] = ∂q∂x
                    if (min_∂ₓq_bc ≤ abs(∂q∂x) ≤ max_∂ₓq_bc)
                        _idx_map[size(_A1, 1)+ctr] = (k, 0, j)
                        ctr += 1
                    end
                end
            end
        end
        _A2 = _A2[1:ctr-1, :]
        _b2 = _b2[1:ctr-1]
        _A = (Matrix ∘ BlockDiagonal)([_A1, _A2])
        _b = vcat(_b1, _b2)
        @test A ≈ _A
        @test b ≈ _b
        @test idx_map == _idx_map
        @test all(j -> 14.0 ≤ esol[1].t[j] ≤ 28.0, time_indices)
        @test all(j -> (esol[1].t[j] < 14.0 || esol[1].t[j] > 28.0), setdiff(1:length(esol[1]), time_indices))
    end

    # Fixing diffusion and reaction 
    begin
        all_q = Float64[]
        all_∂ₜq = Float64[]
        all_∂ₓq = Float64[]
        all_∂²ₓq² = Float64[]
        all_∂ₓq_bc = Float64[]
        all_dLdt = Float64[]
        for k in 1:length(esol)
            sol = esol[k]
            for j in 2:length(sol)
                for i in 1:length(sol[j])
                    push!(all_q, EQL.cell_density(sol, i, j))
                    push!(all_∂ₜq, EQL.cell_∂q∂t(sol, i, j))
                    push!(all_∂ₓq, EQL.cell_∂q∂x(sol, i, j))
                    push!(all_∂²ₓq², EQL.cell_∂²q∂x²(sol, i, j))
                end
                push!(all_∂ₓq_bc, EQL.cell_∂q∂x(sol, length(sol[j]), j))
                push!(all_dLdt, EQL.cell_dLdt(sol, j))
            end
        end
        min_q, max_q = quantile(all_q, [0.0, 1])
        min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [1e-4, 1 - 1e-4])
        min_∂ₓq, max_∂ₓq = quantile(abs.(all_∂ₓq), [1e-2, 1 - 1e-2])
        min_∂²ₓq², max_∂²ₓq² = quantile(abs.(all_∂²ₓq²), [0.3, 1 - 0.3])
        min_∂ₓq_bc, max_∂ₓq_bc = quantile(abs.(all_∂ₓq_bc), [1e-2, 1 - 1e-2])
        min_dLdt, max_dLdt = quantile(abs.(all_dLdt), [1e-5, 1 - 1e-5])
        A, b, idx_map, valid_time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
            diffusion_basis, diffusion_parameters, diffusion_theta=[1.1, 0.9, 0.5],
            reaction_basis, reaction_parameters, reaction_theta=[0.2, 0.6, 1.0],
            rhs_basis, rhs_parameters, average=Val(false),
            moving_boundary_basis, moving_boundary_parameters,
            threshold_tol=(q=0.0, dt=1e-4, dx=1e-2, dx2=0.3, dx_bc=1e-2, dL=1e-5))
        @test esol === _sol
        _neqs = sum(sum(length.(sol.u)) for sol in esol)
        _idx_map = Bijection{Int,NTuple{3,Int}}()
        _A2 = zeros(sum(length.(esol.u)), 2)
        _b2 = zeros(sum(length.(esol.u)))
        global ctr = 1
        for k in 1:length(esol)
            sol = esol[k]
            for j in 2:length(sol)
                global ctr
                q = EQL.cell_density(sol, length(sol.u[j]), j)
                ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
                dLdt = EQL.cell_dLdt(sol, j)
                _A2[ctr, 1] = q^2
                _A2[ctr, 2] = q^3
                _b2[ctr] = ∂q∂x
                if (min_∂ₓq_bc ≤ abs(∂q∂x) ≤ max_∂ₓq_bc)
                    _idx_map[ctr] = (k, 0, j)
                    ctr += 1
                end
            end
        end
        _A2 = _A2[1:ctr-1, :]
        _b2 = _b2[1:ctr-1]
        _A3 = zeros(sum(length.(esol.u)), 3)
        _b3 = zeros(sum(length.(esol.u)))
        global ctr = 1
        for k in 1:length(esol)
            sol = esol[k]
            for j in 2:length(sol)
                global ctr
                q = EQL.cell_density(sol, length(sol.u[j]), j)
                ∂q∂x = EQL.cell_∂q∂x(sol, length(sol.u[j]), j)
                dLdt = EQL.cell_dLdt(sol, j)
                _A3[ctr, 1] = -1 / q * ∂q∂x
                _A3[ctr, 2] = -1 / q^2 * ∂q∂x
                _A3[ctr, 3] = -1 / q^3 * ∂q∂x
                _b3[ctr] = q * dLdt
                if (min_dLdt ≤ abs(dLdt) ≤ max_dLdt)
                    _idx_map[size(_A2, 1)+ctr] = (k, -1, j)
                    ctr += 1
                end
            end
        end
        _A3 = _A3[1:ctr-1, :]
        _b3 = _b3[1:ctr-1]
        _A = (Matrix ∘ BlockDiagonal)([_A2, _A3])
        _b = vcat(_b2, _b3)
        @test A ≈ _A
        @test b ≈ _b
        @test idx_map == _idx_map
        @test valid_time_indices == 2:length(esol[1])
    end

    # Fixing RHS and moving boundary parameters
    begin
        all_q = Float64[]
        all_∂ₜq = Float64[]
        all_∂ₓq = Float64[]
        all_∂²ₓq² = Float64[]
        all_∂ₓq_bc = Float64[]
        all_dLdt = Float64[]
        for k in 1:length(esol)
            sol = esol[k]
            for j in 2:length(sol)
                for i in 1:length(sol[j])
                    push!(all_q, EQL.cell_density(sol, i, j))
                    push!(all_∂ₜq, EQL.cell_∂q∂t(sol, i, j))
                    push!(all_∂ₓq, EQL.cell_∂q∂x(sol, i, j))
                    push!(all_∂²ₓq², EQL.cell_∂²q∂x²(sol, i, j))
                end
                push!(all_∂ₓq_bc, EQL.cell_∂q∂x(sol, length(sol[j]), j))
                push!(all_dLdt, EQL.cell_dLdt(sol, j))
            end
        end
        min_q, max_q = quantile(all_q, [0.0, 1])
        min_∂ₜq, max_∂ₜq = quantile(abs.(all_∂ₜq), [1e-4, 1 - 1e-4])
        min_∂ₓq, max_∂ₓq = quantile(abs.(all_∂ₓq), [1e-2, 1 - 1e-2])
        min_∂²ₓq², max_∂²ₓq² = quantile(abs.(all_∂²ₓq²), [0.3, 1 - 0.3])
        A, b, idx_map, valid_time_indices, qmin, qmax, _sol = EQL.build_basis_system(esol;
            diffusion_basis, diffusion_parameters,
            reaction_basis, reaction_parameters, rhs_theta=[2.0, -2resting_spring_length], moving_boundary_theta=[0.5, spring_constant, 1.2],
            rhs_basis, rhs_parameters, average=Val(false),
            moving_boundary_basis, moving_boundary_parameters,
            threshold_tol=(q=0.0, dt=1e-4, dx=1e-2, dx2=0.3, dx_bc=1e-2, dL=1e-5))
        @test esol === _sol
        _neqs = sum(sum(length.(sol.u)) for sol in esol)
        _A1 = zeros(_neqs, 6)
        _b1 = zeros(_neqs)
        _idx_map = Bijection{Int,NTuple{3,Int}}()
        global ctr = 1
        for k in 1:length(esol)
            for j in 2:length(esol[k])
                for i in 1:length(esol[k][j])
                    global ctr
                    q = EQL.cell_density(esol[k], i, j)
                    ∂q∂x = EQL.cell_∂q∂x(esol[k], i, j)
                    ∂q∂t = EQL.cell_∂q∂t(esol[k], i, j)
                    ∂²q∂x² = EQL.cell_∂²q∂x²(esol[k], i, j)
                    _A1[ctr, 1] = -diffusion_parameters / q^2 * ∂q∂x^2 + diffusion_parameters / q * ∂²q∂x²
                    _A1[ctr, 2] = -2diffusion_parameters / q^3 * ∂q∂x^2 + diffusion_parameters / q^2 * ∂²q∂x²
                    _A1[ctr, 3] = -3diffusion_parameters / q^4 * ∂q∂x^2 + diffusion_parameters / q^3 * ∂²q∂x²
                    _A1[ctr, 4] = Gp.β * q
                    _A1[ctr, 5] = Gp.β * q^2
                    _A1[ctr, 6] = Gp.β * q^3
                    _b1[ctr] = ∂q∂t
                    if (min_q ≤ q ≤ max_q) && (min_∂ₜq ≤ abs(∂q∂t) ≤ max_∂ₜq) && (min_∂ₓq ≤ abs(∂q∂x) ≤ max_∂ₓq) && (min_∂²ₓq² ≤ abs(∂²q∂x²) ≤ max_∂²ₓq²)
                        _idx_map[ctr] = (k, i, j)
                        ctr += 1
                    end
                end
            end
        end
        _A1 = _A1[1:ctr-1, :]
        _b1 = _b1[1:ctr-1]
        _A = (Matrix ∘ BlockDiagonal)([_A1])
        _b = vcat(_b1)
        @test A ≈ _A
        @test b ≈ _b
        @test idx_map == _idx_map
        @test valid_time_indices == 2:length(esol[1])
    end
end