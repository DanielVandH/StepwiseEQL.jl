using ..StepwiseEQL
using EpithelialDynamics1D
using OrdinaryDiffEq
using LinearAlgebra
using Bijections
using ElasticArrays
using DataInterpolations
using Random
using FiniteVolumeMethod1D
using LinearSolve
using Setfield
using StatsBase
using StableRNGs
const EQL = StepwiseEQL

@testset "Projected solve" begin
    for i in 1:100
        n = rand(20:50000)
        m = rand(1:19)
        A = rand(n, m)
        b = rand(n)
        indicators = rand(Bool, m)
        x = A[:, indicators] \ b
        _x = zeros(m)
        _x[indicators] .= x
        __x = EQL.projected_solve(A, b, indicators)
        i == 1 && @inferred EQL.projected_solve(A, b, indicators)
        @test A * _x ≈ A[:, indicators] * x ≈ A * __x
        @test all(iszero, __x[.!indicators])
        @test !any(iszero, __x[indicators])
    end
end

@testset "Training and test subsets" begin
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
        model = EQL.EQLModel(sol;
            diffusion_basis, diffusion_parameters)

        rng = StableRNG(123)
        nt = length(model.cell_sol)
        _training_times = sample(rng, 2:nt, ceil(Int, 0.8(nt - 1)), replace=false, ordered=true)
        _training_subset = Int[]
        for j in _training_times
            r = model.cell_sol.u[j]
            for i in eachindex(r)
                if (1, i, j) ∈ model.idx_map.range
                    push!(_training_subset, model.idx_map((1, i, j)))
                end
            end
        end
        _test_subset = setdiff(1:length(model.b), _training_subset)
        (training_times, training_subset), (test_times, test_subset), pde_times = EQL.get_training_and_test_subsets(model; cross_validation=true, rng=StableRNG(123))
        @test training_subset == _training_subset
        @test test_subset == _test_subset
        @test training_times == _training_times
        @test test_times == setdiff(model.valid_time_indices, _training_times)
        @test pde_times == test_times
        @test issorted(training_subset)
        @test issorted(test_subset) # Having sorted indices helps with contiguous array access
        (training_times, training_subset), (test_times, test_subset), pde_times = EQL.get_training_and_test_subsets(model; cross_validation=false)
        @test training_subset == test_subset == 1:size(model.A, 1)
        @test training_times == test_times == model.valid_time_indices
        @test pde_times == test_times
    end

    @testset "Reaction" begin
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
        esol = solve(ens_prob, Tsit5(); trajectories=10, saveat=0.01)
        sol = esol[1]
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
        model = EQL.EQLModel(esol;
            diffusion_basis, diffusion_parameters,
            reaction_parameters, reaction_basis, average=Val(false),
            time_range=(10, 40))

        rng = StableRNG(123)
        nt = count(10.0 .≤ model.cell_sol[1].t .≤ 40.0)
        ns = length(model.cell_sol)
        _training_times = sample(rng, findall(10.0 .≤ model.cell_sol[1].t .≤ 40.0), ceil(Int, 0.8nt), replace=false, ordered=true)
        _training_subset = Int[]
        for k in 1:ns
            for j in _training_times
                r = model.cell_sol.u[k][j]
                for i in eachindex(r)
                    if (k, i, j) ∈ model.idx_map.range
                        push!(_training_subset, model.idx_map((k, i, j)))
                    end
                end
            end
        end
        _test_subset = setdiff(1:length(model.b), _training_subset)
        (training_times, training_subset), (test_times, test_subset), pde_times = EQL.get_training_and_test_subsets(model; cross_validation=true, rng=StableRNG(123))
        @test training_subset == _training_subset
        @test test_subset == _test_subset
        @test training_times == _training_times
        @test test_times == setdiff(model.valid_time_indices, _training_times)
        @test pde_times == test_times
        @test issorted(training_subset)
        @test issorted(test_subset) # Having sorted indices helps with contiguous array access
        (training_times, training_subset), (test_times, test_subset), pde_times = EQL.get_training_and_test_subsets(model; cross_validation=false)
        @test training_subset == test_subset == 1:size(model.A, 1)
        @test training_times == test_times == model.valid_time_indices
        @test pde_times == test_times

        # with extrapolate_pde 
        rng = StableRNG(123)
        nt = count(10.0 .≤ model.cell_sol[1].t .≤ 40.0)
        ns = length(model.cell_sol)
        _training_times = sample(rng, findall(10.0 .≤ model.cell_sol[1].t .≤ 40.0), ceil(Int, 0.8nt), replace=false, ordered=true)
        _training_subset = Int[]
        for k in 1:ns
            for j in _training_times
                r = model.cell_sol.u[k][j]
                for i in eachindex(r)
                    if (k, i, j) ∈ model.idx_map.range
                        push!(_training_subset, model.idx_map((k, i, j)))
                    end
                end
            end
        end
        _test_subset = setdiff(1:length(model.b), _training_subset)
        _pde_times = setdiff(2:length(model.cell_sol[1].t), sample(rng, 2:length(model.cell_sol[1].t), ceil(Int, 0.8 * length(model.cell_sol[1].t)), replace=false, ordered=true))
        (training_times, training_subset), (test_times, test_subset), pde_times = EQL.get_training_and_test_subsets(model; cross_validation=true, rng=StableRNG(123), extrapolate_pde=true)
        @test training_subset == _training_subset
        @test test_subset == _test_subset
        @test training_times == _training_times
        @test test_times == setdiff(model.valid_time_indices, _training_times)
        @test pde_times == _pde_times
        @test issorted(training_subset)
        @test issorted(test_subset) # Having sorted indices helps with contiguous array access
        @test issorted(pde_times)
        (training_times, training_subset), (test_times, test_subset), pde_times = EQL.get_training_and_test_subsets(model; cross_validation=false, extrapolate_pde=true)
        @test training_subset == test_subset == 1:size(model.A, 1)
        @test training_times == test_times == model.valid_time_indices
        @test pde_times == 2:length(model.cell_sol[1].t)
    end

    @testset "Moving Boundary with no Proliferation" begin
        final_time = 50.0
        domain_length = 30.0
        midpoint = domain_length / 2
        cell_nodes = [LinRange(0, midpoint, 10); LinRange(midpoint, domain_length, 20)] |> unique!
        damping_constant = 1.0
        resting_spring_length = 1.05
        spring_constant = 23.0
        k = spring_constant
        force_law_parameters = (s=resting_spring_length, k=spring_constant)
        force_law = (δ, p) -> p.k * (p.s - δ)
        prob = CellProblem(;
            final_time,
            initial_condition=cell_nodes,
            damping_constant,
            force_law,
            force_law_parameters,
            fix_right=false)
        sol = solve(prob, Tsit5(), saveat=0.1)

        diffusion_basis = BasisSet(
            (u, k) -> k * inv(u),
            (u, k) -> k * inv(u^2),
            (u, k) -> k * inv(u^3)
        )
        rhs_basis = BasisSet(
            (u, s) -> s * u,
            (u, s) -> s * u^2,
            (u, s) -> s * u^3,
            (u, s) -> s * u^4
        )
        moving_boundary_basis = BasisSet(
            (u, k) -> k * inv(u^2),
            (u, k) -> k * inv(u^3),
            (u, k) -> k * inv(u^4)
        )
        diffusion_parameters = spring_constant
        rhs_parameters = resting_spring_length
        moving_boundary_parameters = spring_constant
        model = EQL.EQLModel(sol; mesh_points=500, time_range=(2, 37), diffusion_basis, diffusion_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters)

        rng = StableRNG(123)
        nt = length(model.valid_time_indices)
        _training_times = sample(rng, model.valid_time_indices, ceil(Int, 0.8nt), replace=false, ordered=true)
        _training_subset = Int[]
        for j in _training_times
            r = model.cell_sol.u[j]
            for i in eachindex(r)
                if (1, i, j) ∈ model.idx_map.range
                    push!(_training_subset, model.idx_map((1, i, j)))
                end
            end
            if (1, 0, j) ∈ model.idx_map.range
                push!(_training_subset, model.idx_map((1, 0, j)))
            end
            if (1, -1, j) ∈ model.idx_map.range
                push!(_training_subset, model.idx_map((1, -1, j)))
            end
        end
        sort!(_training_subset)
        _test_subset = setdiff(1:length(model.b), _training_subset)
        (training_times, training_subset), (test_times, test_subset) = EQL.get_training_and_test_subsets(model; cross_validation=true, rng=StableRNG(123))
        @test training_subset == _training_subset
        @test test_subset == _test_subset
        @test training_times == _training_times
        @test test_times == setdiff(model.valid_time_indices, _training_times)
        @test issorted(training_subset)
        @test issorted(test_subset) # Having sorted indices helps with contiguous array access
        (training_times, training_subset), (test_times, test_subset) = EQL.get_training_and_test_subsets(model; cross_validation=false)
        @test training_subset == test_subset == 1:size(model.A, 1)
        @test training_times == test_times == model.valid_time_indices
    end

    @testset "Moving Boundary with Proliferation without Averaging" begin
        final_time = 50.0
        domain_length = 30.0
        midpoint = domain_length / 2
        cell_nodes = [LinRange(0, midpoint, 10); LinRange(midpoint, domain_length, 20)] |> unique!
        damping_constant = 1.0
        resting_spring_length = 1.05
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
        ensemble_sol = solve(ens_prob, Tsit5(); trajectories=10, saveat=1.0)
        single_sol = ensemble_sol[1]

        diffusion_basis = BasisSet(
            (u, k) -> k * inv(u),
            (u, k) -> k * inv(u^2),
            (u, k) -> k * inv(u^3)
        )
        reaction_basis = BasisSet(
            (u, β) -> β * u,
            (u, β) -> β * u^2,
            (u, β) -> β * u^3,
        )
        rhs_basis = BasisSet(
            (u, s) -> s * u,
            (u, s) -> s * u^2,
            (u, s) -> s * u^3,
            (u, s) -> s * u^4
        )
        moving_boundary_basis = BasisSet(
            (u, k) -> k * inv(u^2),
            (u, k) -> k * inv(u^3),
            (u, k) -> k * inv(u^4)
        )
        diffusion_parameters = k
        reaction_parameters = β
        rhs_parameters = resting_spring_length
        moving_boundary_parameters = spring_constant
        model = EQL.EQLModel(ensemble_sol; mesh_points=500, time_range=(10, 40), average=Val(false), diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters)
        rng = StableRNG(123)
        nt = length(model.valid_time_indices)
        ns = length(model.cell_sol)
        _training_times = sample(rng, model.valid_time_indices, ceil(Int, 0.8nt), replace=false, ordered=true)
        _training_subset = Int[]
        for k in 1:ns
            for j in _training_times
                r = model.cell_sol.u[k][j]
                for i in eachindex(r)
                    if (k, i, j) ∈ model.idx_map.range
                        push!(_training_subset, model.idx_map((k, i, j)))
                    end
                end
                if (k, 0, j) ∈ model.idx_map.range
                    push!(_training_subset, model.idx_map((k, 0, j)))
                end
                if (k, -1, j) ∈ model.idx_map.range
                    push!(_training_subset, model.idx_map((k, -1, j)))
                end
            end
        end
        sort!(_training_subset)
        _test_subset = setdiff(1:length(model.b), _training_subset)
        (training_times, training_subset), (test_times, test_subset) = EQL.get_training_and_test_subsets(model; cross_validation=true, rng=StableRNG(123))
        @test training_subset == _training_subset
        @test test_subset == _test_subset
        @test training_times == _training_times
        @test issorted(training_subset)
        @test issorted(test_subset) # Having sorted indices helps with contiguous array access
        (training_times, training_subset), (test_times, test_subset) = EQL.get_training_and_test_subsets(model; cross_validation=false)
        @test training_subset == test_subset == 1:size(model.A, 1)
        @test training_times == test_times == model.valid_time_indices
    end

    @testset "Moving Boundary with Proliferation with Averaging" begin
        final_time = 50.0
        domain_length = 30.0
        midpoint = domain_length / 2
        cell_nodes = [LinRange(0, midpoint, 10); LinRange(midpoint, domain_length, 20)] |> unique!
        damping_constant = 1.0
        resting_spring_length = 1.05
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
        ensemble_sol = solve(ens_prob, Tsit5(); trajectories=10, saveat=1.0)
        single_sol = ensemble_sol[1]

        diffusion_basis = BasisSet(
            (u, k) -> k * inv(u),
            (u, k) -> k * inv(u^2),
            (u, k) -> k * inv(u^3)
        )
        reaction_basis = BasisSet(
            (u, β) -> β * u,
            (u, β) -> β * u^2,
            (u, β) -> β * u^3,
        )
        rhs_basis = BasisSet(
            (u, s) -> s * u,
            (u, s) -> s * u^2,
            (u, s) -> s * u^3,
            (u, s) -> s * u^4
        )
        moving_boundary_basis = BasisSet(
            (u, k) -> k * inv(u^2),
            (u, k) -> k * inv(u^3),
            (u, k) -> k * inv(u^4)
        )
        diffusion_parameters = k
        reaction_parameters = β
        rhs_parameters = resting_spring_length
        moving_boundary_parameters = spring_constant
        model = EQL.EQLModel(ensemble_sol; mesh_points=500, time_range=(10, 40), diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters)
        rng = StableRNG(123)
        nt = length(model.valid_time_indices)
        ns = length(model.cell_sol)
        _training_times = sample(rng, model.valid_time_indices, ceil(Int, 0.8nt), replace=false, ordered=true)
        _training_subset = Int[]
        for j in _training_times
            r = model.cell_sol.u[j]
            for i in eachindex(r)
                if (1, i, j) ∈ model.idx_map.range
                    push!(_training_subset, model.idx_map((1, i, j)))
                end
            end
            if (1, 0, j) ∈ model.idx_map.range
                push!(_training_subset, model.idx_map((1, 0, j)))
            end
            if (1, -1, j) ∈ model.idx_map.range
                push!(_training_subset, model.idx_map((1, -1, j)))
            end
        end
        sort!(_training_subset)
        _test_subset = setdiff(1:length(model.b), _training_subset)
        (training_times, training_subset), (test_times, test_subset) = EQL.get_training_and_test_subsets(model; cross_validation=true, rng=StableRNG(123))
        @test training_subset == _training_subset
        @test test_subset == _test_subset
        @test training_times == _training_times
        @test issorted(training_subset)
        @test issorted(test_subset) # Having sorted indices helps with contiguous array access
        (training_times, training_subset), (test_times, test_subset) = EQL.get_training_and_test_subsets(model; cross_validation=false)
        @test training_subset == test_subset == 1:size(model.A, 1)
        @test training_times == test_times == model.valid_time_indices
    end
end

@testset "Check negative diffuson" begin
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
        (u, p) -> 10inv(u),
        (u, p) -> 10inv(u^2),
        (u, p) -> 10inv(u^3)
    )
    diffusion_parameters = nothing
    model = EQL.EQLModel(sol;
        diffusion_basis, diffusion_parameters)
    nc = 250
    qk = LinRange(extrema(stack(node_densities.(sol.u)))..., nc)
    for i in 1:500
        θ = randn(3)
        flag = any(q -> 10 * θ[1] / q + 10 * θ[2] / q^2 + 10 * θ[3] / q^3 < 0, qk)
        _flag = EQL.check_negative_diffusion_or_moving_boundary(model, θ, nc, false)
        @test flag == _flag
    end
end

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
    model = EQL.EQLModel(sol;
        diffusion_basis, diffusion_parameters)
    model.indicators[2] = false
    rng = StableRNG(9881)
    for cross_validation ∈ (true, false)
        (training_time, training_subset), (test_times, test_subset) = EQL.get_training_and_test_subsets(model; cross_validation, rng=deepcopy(rng))
        A_train = model.A[training_subset, model.indicators]
        b_train = model.b[training_subset]
        θ = A_train \ b_train
        A_test = model.A[test_subset, model.indicators]
        b_test = model.b[test_subset]
        regression_loss = norm(A_test * θ - b_test)^2 / norm(b_test)^2
        _pde = FVMProblem(prob, 1000;
            diffusion_function=diffusion_basis,
            diffusion_parameters=EQL.Parameters(θ=[θ[1], 0.0, θ[2]], p=nothing),
            proliferation=false)
        _pde_sol = solve(_pde, TRBDF2(linsolve=KLUFactorization()), saveat=Δt)
        density_loss = 0.0
        for j in test_times
            for i in 1:length(initial_condition)
                interp = LinearInterpolation(_pde_sol.u[j], _pde_sol.prob.p.geometry.mesh_points)
                pde_q = interp(sol.u[j][i])
                cell_q = EQL.cell_density(sol, i, j)
                #if (1, i, j) ∈ model.idx_map.range && model.idx_map((1, i, j)) ∈ test_subset
                density_loss += (pde_q - cell_q)^2 / cell_q^2
                #end
            end
        end
        for density in (true, false)
            for regression in (true, false)
                _obj = regression * log(regression_loss / length(test_subset)) + density * log(density_loss / sum(length(u) for u in sol.u[test_times])) + 2
                @test EQL.evaluate_model_loss(model,
                    model.indicators,
                    default_loss(; density, regression);
                    cross_validation, rng=deepcopy(rng)) ≈ _obj
            end
        end
    end
end

@testset "Reaction without Averaging" begin
    final_time = 2.5
    domain_length = 30.0
    midpoint = domain_length / 2
    cell_nodes = [LinRange(0, midpoint, 60); LinRange(midpoint, domain_length, 60)] |> unique!
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
    ens_prob = EnsembleProblem(prob; rng=StableRNG(7))
    esol = solve(ens_prob, Tsit5(), EnsembleSerial(); trajectories=50, saveat=0.2)
    sol = esol[1]
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
    reaction_parameters = (Gp.β, Gp.K)
    emodel = EQL.EQLModel(esol, mesh_points=500;
        diffusion_basis, diffusion_parameters,
        reaction_parameters, reaction_basis, average=Val(false),
        threshold_tol=(q=0.01, dt=0.01))
    emodel_fixed_diffusion = EQL.EQLModel(esol, mesh_points=500;
        diffusion_basis, diffusion_parameters,
        reaction_parameters, reaction_basis, average=Val(false),
        diffusion_theta=[0.1, 1.0, 0.3],
        threshold_tol=(q=0.01, dt=0.01))
    emodel_fixed_reaction = EQL.EQLModel(esol;
        diffusion_basis, diffusion_parameters, average=Val(false),
        reaction_parameters, reaction_basis,
        reaction_theta=[1.0, -1.0, 0.02],
        threshold_tol=(q=0.01, dt=0.01))
    model = EQL.EQLModel(esol[15], mesh_points=1000;
        diffusion_basis, diffusion_parameters,
        reaction_parameters, reaction_basis, average=Val(false),
        threshold_tol=(q=0.01, dt=0.01))
    model_fixed_diffusion = EQL.EQLModel(sol, mesh_points=500;
        diffusion_basis, diffusion_parameters, average=Val(false),
        reaction_parameters, reaction_basis,
        diffusion_theta=[0.04, 1.1, 0.005],
        threshold_tol=(q=0.01, dt=0.01))
    model_fixed_reaction = EQL.EQLModel(sol; average=Val(false),
        diffusion_basis, diffusion_parameters,
        reaction_parameters, reaction_basis,
        reaction_theta=[1, -1, 0.0],
        threshold_tol=(q=0.01, dt=0.01))
    emodel.indicators[[3, 6]] .= false
    emodel_fixed_diffusion.indicators[[3]] .= false
    emodel_fixed_reaction.indicators[[1, 3]] .= false
    model.indicators[[1, 3, 6]] .= false
    model_fixed_diffusion.indicators[[1, 3]] .= false
    model_fixed_reaction.indicators[[3]] .= false
    models = (
        emodel, emodel_fixed_diffusion, emodel_fixed_reaction,
        model, model_fixed_diffusion, model_fixed_reaction
    )

    rngs = StableRNG.((123, 561, 1200, 991, 2991, 146767))
    cross_validation = true
    subsets = [EQL.get_training_and_test_subsets(model; cross_validation=cross_validation, rng=deepcopy(rng)) for (model, rng) in zip(models, rngs)]
    training_subsets = first.(subsets)
    test_subsets = getindex.(subsets, 2)
    for s in subsets
        @test s[3] == s[2][1]
    end
    A_trains = [model.A[training_subset[2], model.indicators] for (model, training_subset) in zip(models, training_subsets)]
    b_trains = [model.b[training_subset[2]] for (model, training_subset) in zip(models, training_subsets)]
    A_tests = [model.A[test_subset[2], model.indicators] for (model, test_subset) in zip(models, test_subsets)]
    b_tests = [model.b[test_subset[2]] for (model, test_subset) in zip(models, test_subsets)]
    θs = [A_train \ b_train for (A_train, b_train) in zip(A_trains, b_trains)]
    regression_loss = [(norm(A_test * θ - b_test) / norm(b_test))^2 for (A_test, θ, b_test) in zip(A_tests, θs, b_tests)]
    pdes = [EQL.rebuild_pde(model, EQL.projected_solve(model, training_subset[2])) for (model, training_subset) in zip(models, training_subsets)]
    pde_sols = [solve(pde, TRBDF2(linsolve=KLUFactorization()), saveat=0.2) for pde in pdes]
    density_loss = zero(regression_loss)
    for ℓ in 1:6
        model = models[ℓ]
        _pde = pdes[ℓ]
        _pde_sol = pde_sols[ℓ]
        cell_sol = model.cell_sol
        if cell_sol isa EnsembleSolution
            for k in 1:length(cell_sol)
                for j in test_subsets[ℓ][1]
                    for i in 1:length(cell_sol.u[k][j])
                        interp = LinearInterpolation(_pde_sol.u[j], _pde_sol.prob.p.geometry.mesh_points)
                        pde_q = interp(cell_sol.u[k][j][i])
                        cell_q = EQL.cell_density(cell_sol.u[k], i, j)
                        #if (k, i, j) ∈ model.idx_map.range && model.idx_map((k, i, j)) ∈ test_subsets[ℓ]
                        density_loss[ℓ] += (pde_q - cell_q)^2 / cell_q^2
                        #end
                    end
                end
            end
        else
            for j in test_subsets[ℓ][1]
                for i in 1:length(cell_sol[j])
                    interp = LinearInterpolation(_pde_sol.u[j], _pde_sol.prob.p.geometry.mesh_points)
                    pde_q = interp(cell_sol[j][i])
                    cell_q = EQL.cell_density(cell_sol, i, j)
                    #if (1, i, j) ∈ model.idx_map.range && model.idx_map((1, i, j)) ∈ test_subsets[ℓ]
                    density_loss[ℓ] += (pde_q - cell_q)^2 / cell_q^2
                    #end
                end
            end
        end
    end
    for (model,
        test_subset,
        regression_loss,
        density_loss,
        rng) in zip(
        models,
        test_subsets,
        regression_loss,
        density_loss,
        rngs)
        _nt = 0
        _test_times, _test_subset = test_subset
        if model.cell_sol isa EnsembleSolution
            for k in model.simulation_indices
                for j in _test_times
                    for i in eachindex(model.cell_sol.u[k][j])
                        _nt += 1
                    end
                end
            end
        else
            for j in _test_times
                for i in eachindex(model.cell_sol.u[j])
                    _nt += 1
                end
            end
        end
        for density in (true, false)
            for regression in (true, false)
                loss = regression * log(regression_loss / length(_test_subset)) +
                       density * log(density_loss / _nt) +
                       count(model.indicators)
                _obj = EQL.evaluate_model_loss(model, model.indicators, default_loss(; density, regression); cross_validation, rng=deepcopy(rng), num_constraint_checks=0)
                @test loss ≈ _obj
                @inferred EQL.evaluate_model_loss(model, model.indicators, default_loss(; density, regression); cross_validation, rng=deepcopy(rng), num_constraint_checks=0)
            end
        end
    end
end

@testset "Reaction with Averaging" begin
    final_time = 2.5
    domain_length = 30.0
    midpoint = domain_length / 2
    cell_nodes = [LinRange(0, midpoint, 60); LinRange(midpoint, domain_length, 60)] |> unique!
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
    ens_prob = EnsembleProblem(prob; rng=StableRNG(7))
    esol = solve(ens_prob, Tsit5(), EnsembleSerial(); trajectories=50, saveat=0.2)
    sol = esol[1]
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
    reaction_parameters = (Gp.β, Gp.K)
    emodel = EQL.EQLModel(esol, mesh_points=500;
        diffusion_basis, diffusion_parameters,
        reaction_parameters, reaction_basis, average=Val(true),
        threshold_tol=(q=0.01, dt=0.01))
    emodel_fixed_diffusion = EQL.EQLModel(esol, mesh_points=500;
        diffusion_basis, diffusion_parameters,
        reaction_parameters, reaction_basis, average=Val(true),
        diffusion_theta=[0.1, 1.0, 0.3],
        threshold_tol=(q=0.01, dt=0.01))
    emodel_fixed_reaction = EQL.EQLModel(esol;
        diffusion_basis, diffusion_parameters, average=Val(true),
        reaction_parameters, reaction_basis,
        reaction_theta=[1.0, -1.0, 0.02],
        threshold_tol=(q=0.01, dt=0.01))
    model = EQL.EQLModel(esol[15], mesh_points=1000;
        diffusion_basis, diffusion_parameters,
        reaction_parameters, reaction_basis,
        threshold_tol=(q=0.01, dt=0.01))
    model_fixed_diffusion = EQL.EQLModel(sol, mesh_points=500;
        diffusion_basis, diffusion_parameters,
        reaction_parameters, reaction_basis,
        diffusion_theta=[0.04, 1.1, 0.005],
        threshold_tol=(q=0.01, dt=0.01))
    model_fixed_reaction = EQL.EQLModel(sol;
        diffusion_basis, diffusion_parameters,
        reaction_parameters, reaction_basis,
        reaction_theta=[1, -1, 0.0],
        threshold_tol=(q=0.01, dt=0.01))
    emodel.indicators[[3, 6]] .= false
    emodel_fixed_diffusion.indicators[[3]] .= false
    emodel_fixed_reaction.indicators[[1, 3]] .= false
    model.indicators[[1, 3, 6]] .= false
    model_fixed_diffusion.indicators[[1, 3]] .= false
    model_fixed_reaction.indicators[[3]] .= false
    models = (
        emodel, emodel_fixed_diffusion, emodel_fixed_reaction,
        model, model_fixed_diffusion, model_fixed_reaction
    )

    rngs = StableRNG.((123, 561, 1200, 991, 2991, 146767))
    cross_validation = true
    subsets = [EQL.get_training_and_test_subsets(model; cross_validation=cross_validation, rng=deepcopy(rng)) for (model, rng) in zip(models, rngs)]
    training_subsets = first.(subsets)
    test_subsets = getindex.(subsets, 2)
    for s in subsets
        @test s[3] == s[2][1]
    end
    A_trains = [model.A[training_subset[2], model.indicators] for (model, training_subset) in zip(models, training_subsets)]
    b_trains = [model.b[training_subset[2]] for (model, training_subset) in zip(models, training_subsets)]
    A_tests = [model.A[test_subset[2], model.indicators] for (model, test_subset) in zip(models, test_subsets)]
    b_tests = [model.b[test_subset[2]] for (model, test_subset) in zip(models, test_subsets)]
    θs = [A_train \ b_train for (A_train, b_train) in zip(A_trains, b_trains)]
    regression_loss = [(norm(A_test * θ - b_test) / norm(b_test))^2 for (A_test, θ, b_test) in zip(A_tests, θs, b_tests)]
    pdes = [EQL.rebuild_pde(model, EQL.projected_solve(model, training_subset[2])) for (model, training_subset) in zip(models, training_subsets)]
    pde_sols = [solve(pde, TRBDF2(linsolve=KLUFactorization()), saveat=0.2) for pde in pdes]
    density_loss = zero(regression_loss)
    for ℓ in 1:6
        model = models[ℓ]
        _pde = pdes[ℓ]
        _pde_sol = pde_sols[ℓ]
        cell_sol = model.cell_sol
        if cell_sol isa EnsembleSolution
            for k in 1:length(cell_sol)
                for j in test_subsets[ℓ][1]
                    for i in 1:length(cell_sol.u[k][j])
                        interp = LinearInterpolation(_pde_sol.u[j], _pde_sol.prob.p.geometry.mesh_points)
                        pde_q = interp(cell_sol.u[k][j][i])
                        cell_q = EQL.cell_density(cell_sol.u[k], i, j)
                        #if (k, i, j) ∈ model.idx_map.range && model.idx_map((k, i, j)) ∈ test_subsets[ℓ]
                        density_loss[ℓ] += (pde_q - cell_q)^2 / cell_q^2
                        #end
                    end
                end
            end
        else
            for j in test_subsets[ℓ][1]
                for i in 1:length(cell_sol.u[j])
                    interp = LinearInterpolation(_pde_sol.u[j], _pde_sol.prob.p.geometry.mesh_points)
                    pde_q = interp(cell_sol.u[j][i])
                    cell_q = EQL.cell_density(cell_sol, i, j)
                    #if (1, i, j) ∈ model.idx_map.range && model.idx_map((1, i, j)) ∈ test_subsets[ℓ]
                    density_loss[ℓ] += (pde_q - cell_q)^2 / cell_q^2
                    #end
                end
            end
        end
    end
    for (model,
        test_subset,
        regression_loss,
        density_loss,
        rng) in zip(
        models,
        test_subsets,
        regression_loss,
        density_loss,
        rngs)
        _nt = 0
        _test_times, _test_subset = test_subset
        if model.cell_sol isa EnsembleSolution
            for k in model.simulation_indices
                for j in _test_times
                    for i in eachindex(model.cell_sol.u[k][j])
                        _nt += 1
                    end
                end
            end
        else
            for j in _test_times
                for i in eachindex(model.cell_sol.u[j])
                    _nt += 1
                end
            end
        end
        for density in (true, false)
            for regression in (true, false)
                loss = regression * log(regression_loss / length(_test_subset)) +
                       density * log(density_loss / _nt) +
                       count(model.indicators)
                _obj = EQL.evaluate_model_loss(model, model.indicators, default_loss(; density, regression); cross_validation, rng=deepcopy(rng), num_constraint_checks=0)
                @test loss ≈ _obj
                @inferred EQL.evaluate_model_loss(model, model.indicators, default_loss(; density, regression); cross_validation, rng=deepcopy(rng), num_constraint_checks=0)
            end
        end
    end
end

@testset "Moving Boundary with Proliferation" begin
    final_time = 50.0
    domain_length = 30.0
    midpoint = domain_length / 2
    cell_nodes = [LinRange(0, midpoint, 10); LinRange(midpoint, domain_length, 20)] |> unique!
    damping_constant = 1.0
    resting_spring_length = 1.05
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
    ensemble_sol = solve(ens_prob, Tsit5(); trajectories=10, saveat=0.2)
    single_sol = ensemble_sol[1]

    diffusion_basis = BasisSet(
        (u, k) -> k * inv(u),
        (u, k) -> k * inv(u^2),
        (u, k) -> k * inv(u^3)
    )
    reaction_basis = BasisSet(
        (u, β) -> β * K * u,
        (u, β) -> -β * u^2,
        (u, β) -> β * u^3,
    )
    rhs_basis = BasisSet(
        (u, s) -> s * u,
        (u, s) -> 2u^2,
        (u, s) -> -2s * u^3,
        (u, s) -> s * u^4
    )
    moving_boundary_basis = BasisSet(
        (u, k) -> k * inv(u),
        (u, k) -> k * inv(u^2),
        (u, k) -> k * inv(u^3)
    )
    diffusion_parameters = k
    reaction_parameters = β
    rhs_parameters = resting_spring_length
    moving_boundary_parameters = spring_constant
    diffusion_theta = [0.0, 1.0, 0.0]
    reaction_theta = [1.0, 1.0, 0.0]
    rhs_theta = [0.0, 1.0, 1.0, 0.0]
    moving_boundary_theta = [0.0, 1.0, 0.0]

    Id = [false, true, false]
    Ir = [true, true, false]
    Irhs = [true, true, false, false]
    Imb = [false, true, false]
    aggregate_ensemble_model = EQL.EQLModel(ensemble_sol; average=Val(false), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters)
    aggregate_ensemble_model_fixed_diffusion = EQL.EQLModel(ensemble_sol; average=Val(false), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta)
    aggregate_ensemble_model_fixed_reaction = EQL.EQLModel(ensemble_sol; average=Val(false), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta)
    aggregate_ensemble_model_fixed_rhs = EQL.EQLModel(ensemble_sol; average=Val(false), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, rhs_theta)
    aggregate_ensemble_model_fixed_moving_boundary = EQL.EQLModel(ensemble_sol; average=Val(false), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, moving_boundary_theta)
    aggregate_ensemble_model_fixed_diffusion_reaction = EQL.EQLModel(ensemble_sol; average=Val(false), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta)
    aggregate_ensemble_model_fixed_diffusion_rhs = EQL.EQLModel(ensemble_sol; average=Val(false), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, rhs_theta)
    aggregate_ensemble_model_fixed_diffusion_moving_boundary = EQL.EQLModel(ensemble_sol; average=Val(false), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, moving_boundary_theta)
    aggregate_ensemble_model_fixed_reaction_rhs = EQL.EQLModel(ensemble_sol; average=Val(false), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, rhs_theta)
    aggregate_ensemble_model_fixed_reaction_moving_boundary = EQL.EQLModel(ensemble_sol; average=Val(false), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, moving_boundary_theta)
    aggregate_ensemble_model_fixed_rhs_moving_boundary = EQL.EQLModel(ensemble_sol; average=Val(false), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, rhs_theta, moving_boundary_theta)
    aggregate_ensemble_model_fixed_diffusion_reaction_rhs = EQL.EQLModel(ensemble_sol; average=Val(false), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta, rhs_theta)
    aggregate_ensemble_model_fixed_diffusion_reaction_moving_boundary = EQL.EQLModel(ensemble_sol; average=Val(false), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta, moving_boundary_theta)
    aggregate_ensemble_model_fixed_diffusion_rhs_moving_boundary = EQL.EQLModel(ensemble_sol; average=Val(false), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, rhs_theta, moving_boundary_theta)
    aggregate_ensemble_model_fixed_reaction_rhs_moving_boundary = EQL.EQLModel(ensemble_sol; average=Val(false), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, rhs_theta, moving_boundary_theta)
    single_model = EQL.EQLModel(single_sol; mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters)
    single_model_fixed_diffusion = EQL.EQLModel(single_sol; mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta)
    single_model_fixed_reaction = EQL.EQLModel(single_sol; mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta)
    single_model_fixed_rhs = EQL.EQLModel(single_sol; mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, rhs_theta)
    single_model_fixed_moving_boundary = EQL.EQLModel(single_sol; mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, moving_boundary_theta)
    single_model_fixed_diffusion_reaction = EQL.EQLModel(single_sol; mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta)
    single_model_fixed_diffusion_rhs = EQL.EQLModel(single_sol; mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, rhs_theta)
    single_model_fixed_diffusion_moving_boundary = EQL.EQLModel(single_sol; mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, moving_boundary_theta)
    single_model_fixed_reaction_rhs = EQL.EQLModel(single_sol; mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, rhs_theta)
    single_model_fixed_reaction_moving_boundary = EQL.EQLModel(single_sol; mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, moving_boundary_theta)
    single_model_fixed_rhs_moving_boundary = EQL.EQLModel(single_sol; mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, rhs_theta, moving_boundary_theta)
    single_model_fixed_diffusion_reaction_rhs = EQL.EQLModel(single_sol; mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta, rhs_theta)
    single_model_fixed_diffusion_reaction_moving_boundary = EQL.EQLModel(single_sol; mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta, moving_boundary_theta)
    single_model_fixed_diffusion_rhs_moving_boundary = EQL.EQLModel(single_sol; mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, rhs_theta, moving_boundary_theta)
    single_model_fixed_reaction_rhs_moving_boundary = EQL.EQLModel(single_sol; mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, rhs_theta, moving_boundary_theta)
    average_ensemble_model = EQL.EQLModel(ensemble_sol; average=Val(true), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters)
    average_ensemble_model_fixed_diffusion = EQL.EQLModel(ensemble_sol; average=Val(true), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta)
    average_ensemble_model_fixed_reaction = EQL.EQLModel(ensemble_sol; average=Val(true), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta)
    average_ensemble_model_fixed_rhs = EQL.EQLModel(ensemble_sol; average=Val(true), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, rhs_theta)
    average_ensemble_model_fixed_moving_boundary = EQL.EQLModel(ensemble_sol; average=Val(true), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, moving_boundary_theta)
    average_ensemble_model_fixed_diffusion_reaction = EQL.EQLModel(ensemble_sol; average=Val(true), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta)
    average_ensemble_model_fixed_diffusion_rhs = EQL.EQLModel(ensemble_sol; average=Val(true), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, rhs_theta)
    average_ensemble_model_fixed_diffusion_moving_boundary = EQL.EQLModel(ensemble_sol; average=Val(true), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, moving_boundary_theta)
    average_ensemble_model_fixed_reaction_rhs = EQL.EQLModel(ensemble_sol; average=Val(true), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, rhs_theta)
    average_ensemble_model_fixed_reaction_moving_boundary = EQL.EQLModel(ensemble_sol; average=Val(true), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, moving_boundary_theta)
    average_ensemble_model_fixed_rhs_moving_boundary = EQL.EQLModel(ensemble_sol; average=Val(true), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, rhs_theta, moving_boundary_theta)
    average_ensemble_model_fixed_diffusion_reaction_rhs = EQL.EQLModel(ensemble_sol; average=Val(true), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta, rhs_theta)
    average_ensemble_model_fixed_diffusion_reaction_moving_boundary = EQL.EQLModel(ensemble_sol; average=Val(true), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta, moving_boundary_theta)
    average_ensemble_model_fixed_diffusion_rhs_moving_boundary = EQL.EQLModel(ensemble_sol; average=Val(true), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, rhs_theta, moving_boundary_theta)
    average_ensemble_model_fixed_reaction_rhs_moving_boundary = EQL.EQLModel(ensemble_sol; average=Val(true), mesh_points=50, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, rhs_theta, moving_boundary_theta)
    aggregate_ensemble_model.indicators .= [Id; Ir; Irhs; Imb]
    aggregate_ensemble_model_fixed_diffusion.indicators .= [Ir; Irhs; Imb]
    aggregate_ensemble_model_fixed_reaction.indicators .= [Id; Irhs; Imb]
    aggregate_ensemble_model_fixed_rhs.indicators .= [Id; Ir; Imb]
    aggregate_ensemble_model_fixed_moving_boundary.indicators .= [Id; Ir; Irhs]
    aggregate_ensemble_model_fixed_diffusion_reaction.indicators .= [Irhs; Imb]
    aggregate_ensemble_model_fixed_diffusion_rhs.indicators .= [Ir; Imb]
    aggregate_ensemble_model_fixed_diffusion_moving_boundary.indicators .= [Ir; Irhs]
    aggregate_ensemble_model_fixed_reaction_rhs.indicators .= [Id; Imb]
    aggregate_ensemble_model_fixed_reaction_moving_boundary.indicators .= [Id; Irhs]
    aggregate_ensemble_model_fixed_rhs_moving_boundary.indicators .= [Id; Ir]
    aggregate_ensemble_model_fixed_diffusion_reaction_rhs.indicators .= [Imb;]
    aggregate_ensemble_model_fixed_diffusion_reaction_moving_boundary.indicators .= [Irhs;]
    aggregate_ensemble_model_fixed_diffusion_rhs_moving_boundary.indicators .= [Ir;]
    aggregate_ensemble_model_fixed_reaction_rhs_moving_boundary.indicators .= [Id;]
    single_model.indicators .= [Id; Ir; Irhs; Imb]
    single_model_fixed_diffusion.indicators .= [Ir; Irhs; Imb]
    single_model_fixed_reaction.indicators .= [Id; Irhs; Imb]
    single_model_fixed_rhs.indicators .= [Id; Ir; Imb]
    single_model_fixed_moving_boundary.indicators .= [Id; Ir; Irhs]
    single_model_fixed_diffusion_reaction.indicators .= [Irhs; Imb]
    single_model_fixed_diffusion_rhs.indicators .= [Ir; Imb]
    single_model_fixed_diffusion_moving_boundary.indicators .= [Ir; Irhs]
    single_model_fixed_reaction_rhs.indicators .= [Id; Imb]
    single_model_fixed_reaction_moving_boundary.indicators .= [Id; Irhs]
    single_model_fixed_rhs_moving_boundary.indicators .= [Id; Ir]
    single_model_fixed_diffusion_reaction_rhs.indicators .= [Imb;]
    single_model_fixed_diffusion_reaction_moving_boundary.indicators .= [Irhs;]
    single_model_fixed_diffusion_rhs_moving_boundary.indicators .= [Ir;]
    single_model_fixed_reaction_rhs_moving_boundary.indicators .= [Id;]
    average_ensemble_model.indicators .= [Id; Ir; Irhs; Imb]
    average_ensemble_model_fixed_diffusion.indicators .= [Ir; Irhs; Imb]
    average_ensemble_model_fixed_reaction.indicators .= [Id; Irhs; Imb]
    average_ensemble_model_fixed_rhs.indicators .= [Id; Ir; Imb]
    average_ensemble_model_fixed_moving_boundary.indicators .= [Id; Ir; Irhs]
    average_ensemble_model_fixed_diffusion_reaction.indicators .= [Irhs; Imb]
    average_ensemble_model_fixed_diffusion_rhs.indicators .= [Ir; Imb]
    average_ensemble_model_fixed_diffusion_moving_boundary.indicators .= [Ir; Irhs]
    average_ensemble_model_fixed_reaction_rhs.indicators .= [Id; Imb]
    average_ensemble_model_fixed_reaction_moving_boundary.indicators .= [Id; Irhs]
    average_ensemble_model_fixed_rhs_moving_boundary.indicators .= [Id; Ir]
    average_ensemble_model_fixed_diffusion_reaction_rhs.indicators .= [Imb;]
    average_ensemble_model_fixed_diffusion_reaction_moving_boundary.indicators .= [Irhs;]
    average_ensemble_model_fixed_diffusion_rhs_moving_boundary.indicators .= [Ir;]
    average_ensemble_model_fixed_reaction_rhs_moving_boundary.indicators .= [Id;]
    models = (
        aggregate_ensemble_model,#1
        aggregate_ensemble_model_fixed_diffusion,#2
        aggregate_ensemble_model_fixed_reaction,#3
        aggregate_ensemble_model_fixed_rhs,#4
        aggregate_ensemble_model_fixed_moving_boundary,#5
        aggregate_ensemble_model_fixed_diffusion_reaction,#6
        aggregate_ensemble_model_fixed_diffusion_rhs,#7
        aggregate_ensemble_model_fixed_diffusion_moving_boundary,#8
        aggregate_ensemble_model_fixed_reaction_rhs,#9
        aggregate_ensemble_model_fixed_reaction_moving_boundary,#10
        aggregate_ensemble_model_fixed_rhs_moving_boundary,#11
        aggregate_ensemble_model_fixed_diffusion_reaction_rhs,#12
        aggregate_ensemble_model_fixed_diffusion_reaction_moving_boundary,#13
        aggregate_ensemble_model_fixed_diffusion_rhs_moving_boundary,#14
        aggregate_ensemble_model_fixed_reaction_rhs_moving_boundary,
        single_model,
        single_model_fixed_diffusion,
        single_model_fixed_reaction,
        single_model_fixed_rhs,
        single_model_fixed_moving_boundary,
        single_model_fixed_diffusion_reaction,
        single_model_fixed_diffusion_rhs,
        single_model_fixed_diffusion_moving_boundary,
        single_model_fixed_reaction_rhs,
        single_model_fixed_reaction_moving_boundary,
        single_model_fixed_rhs_moving_boundary,
        single_model_fixed_diffusion_reaction_rhs,
        single_model_fixed_diffusion_reaction_moving_boundary,
        single_model_fixed_diffusion_rhs_moving_boundary,
        single_model_fixed_reaction_rhs_moving_boundary,
        average_ensemble_model,
        average_ensemble_model_fixed_diffusion,
        average_ensemble_model_fixed_reaction,
        average_ensemble_model_fixed_rhs,
        average_ensemble_model_fixed_moving_boundary,
        average_ensemble_model_fixed_diffusion_reaction,
        average_ensemble_model_fixed_diffusion_rhs,
        average_ensemble_model_fixed_diffusion_moving_boundary,
        average_ensemble_model_fixed_reaction_rhs,
        average_ensemble_model_fixed_reaction_moving_boundary,
        average_ensemble_model_fixed_rhs_moving_boundary,
        average_ensemble_model_fixed_diffusion_reaction_rhs,
        average_ensemble_model_fixed_diffusion_reaction_moving_boundary,
        average_ensemble_model_fixed_diffusion_rhs_moving_boundary,
        average_ensemble_model_fixed_reaction_rhs_moving_boundary
    )

    rngs = StableRNG.(1:length(models))
    for cross_validation in (true, false)
        for leading_edge_error in (true, false)
            for extrapolate_pde in (true, false)
                subsets = [EQL.get_training_and_test_subsets(model; extrapolate_pde=extrapolate_pde, cross_validation=cross_validation, rng=deepcopy(rng)) for (model, rng) in zip(models, rngs)]
                training_subsets = first.(subsets)
                test_subsets = getindex.(subsets, 2)
                pde_times = last.(subsets)
                if !extrapolate_pde
                    for s in subsets
                        @test s[3] == s[2][1]
                    end
                end
                A_trains = [model.A[training_subset[2], model.indicators] for (model, training_subset) in zip(models, training_subsets)]
                b_trains = [model.b[training_subset[2]] for (model, training_subset) in zip(models, training_subsets)]
                A_tests = [model.A[test_subset[2], model.indicators] for (model, test_subset) in zip(models, test_subsets)]
                b_tests = [model.b[test_subset[2]] for (model, test_subset) in zip(models, test_subsets)]
                θs = [A_train \ b_train for (A_train, b_train) in zip(A_trains, b_trains)]
                regression_loss = [(norm(A_test * θ - b_test) / norm(b_test))^2 for (A_test, θ, b_test) in zip(A_tests, θs, b_tests)]
                pdes = [EQL.rebuild_pde(model, EQL.projected_solve(model, training_subset[2])) for (model, training_subset) in zip(models, training_subsets)]
                pde_sols = [solve(pde, TRBDF2(linsolve=KLUFactorization()), saveat=0.2) for pde in pdes]
                density_loss = zero(regression_loss)
                leading_edge_loss = zero(regression_loss)
                total_loss = zeros(4, length(density_loss))
                src_loss = zeros(4, length(density_loss))
                Base.Threads.@threads for ℓ in 1:length(models)
                    model = models[ℓ]
                    _pde = pdes[ℓ]
                    _pde_sol = pde_sols[ℓ]
                    density_loss[ℓ] = 0.0
                    _nt = 0
                    _ne = 0
                    if !SciMLBase.successful_retcode(_pde_sol) || any(≤(0), _pde_sol[end, :])
                        density_loss[ℓ] = Inf
                        leading_edge_loss[ℓ] = Inf
                    else
                        if !leading_edge_error
                            cell_sol = model.cell_sol
                            if cell_sol isa EnsembleSolution
                                for k in 1:length(cell_sol)
                                    for j in pde_times[ℓ]
                                        for i in 1:length(cell_sol.u[k][j])
                                            interp = @views LinearInterpolation(_pde_sol.u[j][begin:(end-1)], _pde_sol.prob.p.geometry.mesh_points)
                                            L′ = cell_sol.u[k][j][i] / _pde_sol.u[j][end]
                                            pde_q = interp(L′)
                                            if pde_q < 0 || L′ > 1.0
                                                pde_q = 0.0
                                            end
                                            cell_q = EQL.cell_density(cell_sol.u[k], i, j)
                                            _nt += 1
                                            density_loss[ℓ] += (pde_q - cell_q)^2 / cell_q^2
                                        end
                                    end
                                end
                            else
                                for j in pde_times[ℓ]
                                    for i in 1:length(cell_sol.u[j])
                                        interp = @views LinearInterpolation(_pde_sol.u[j][begin:(end-1)], _pde_sol.prob.p.geometry.mesh_points)
                                        L′ = cell_sol.u[j][i] / _pde_sol.u[j][end]
                                        pde_q = interp(L′)
                                        if pde_q < 0 || L′ > 1.0
                                            pde_q = 0.0
                                        end
                                        cell_q = EQL.cell_density(cell_sol, i, j)
                                        _nt += 1
                                        density_loss[ℓ] += (pde_q - cell_q)^2 / cell_q^2
                                    end
                                end
                            end
                        else
                            cell_sol = model.cell_sol
                            if cell_sol isa EnsembleSolution
                                for k in 1:length(cell_sol)
                                    for j in pde_times[ℓ]
                                        for i in 1:length(cell_sol.u[k][j])
                                            interp = @views LinearInterpolation(_pde_sol.u[j][begin:(end-1)], _pde_sol.prob.p.geometry.mesh_points)
                                            L′ = cell_sol.u[k][j][i] / _pde_sol.u[j][end]
                                            pde_q = interp(L′)
                                            if pde_q < 0 || L′ > 1.0
                                                pde_q = 0.0
                                            end
                                            cell_q = EQL.cell_density(cell_sol.u[k], i, j)
                                            _nt += 1
                                            density_loss[ℓ] += (pde_q - cell_q)^2 / cell_q^2
                                        end
                                        cell_L = cell_sol.u[k][j][end]
                                        pde_L = _pde_sol.u[j][end]
                                        edge_loss = (cell_L - pde_L)^2 / cell_L^2
                                        _ne += 1
                                        leading_edge_loss[ℓ] += edge_loss
                                    end
                                end
                            else
                                for j in pde_times[ℓ]
                                    for i in 1:length(cell_sol.u[j])
                                        interp = @views LinearInterpolation(_pde_sol.u[j][begin:(end-1)], _pde_sol.prob.p.geometry.mesh_points)
                                        L′ = cell_sol.u[j][i] / _pde_sol.u[j][end]
                                        pde_q = interp(L′)
                                        if pde_q < 0 || L′ > 1.0
                                            pde_q = 0.0
                                        end
                                        cell_q = EQL.cell_density(cell_sol, i, j)
                                        _nt += 1
                                        density_loss[ℓ] += (pde_q - cell_q)^2 / cell_q^2
                                    end
                                    cell_L = cell_sol.u[j][end]
                                    pde_L = _pde_sol.u[j][end]
                                    edge_loss = (cell_L - pde_L)^2 / cell_L^2
                                    _ne += 1
                                    leading_edge_loss[ℓ] += edge_loss
                                end
                            end
                        end
                    end
                    ctr = 1
                    for density in (true, false)
                        for regression in (true, false)
                            loss = regression * log(regression_loss[ℓ] / length(test_subsets[ℓ][2])) +
                                   density * log(density_loss[ℓ] / _nt) +
                                   count(model.indicators)
                            if leading_edge_error
                                loss += density * log(leading_edge_loss[ℓ] / _ne)
                            end
                            total_loss[ctr, ℓ] = isfinite(loss) ? loss : Inf
                            src_loss[ctr, ℓ] = EQL.evaluate_model_loss(model, model.indicators, default_loss(; density, regression); cross_validation, rng=deepcopy(rngs[ℓ]), num_constraint_checks=0, leading_edge_error, extrapolate_pde)
                            if !(total_loss[ctr, ℓ] ≈ src_loss[ctr, ℓ])
                                println("Failed combination: density = $density, regression = $regression, leading_edge_error = $leading_edge_error, extrapolate_pde = $extrapolate_pde")
                            end
                            ctr += 1
                            rand() < 0.05 && @inferred EQL.evaluate_model_loss(model, model.indicators, default_loss(; density, regression); cross_validation, rng=deepcopy(rngs[ℓ]), num_constraint_checks=0, leading_edge_error, extrapolate_pde)
                        end
                    end
                end
                for (manual, fnc) in zip(total_loss, src_loss)
                    @test manual ≈ fnc rtol = 1e-1
                end
            end
        end
    end
end