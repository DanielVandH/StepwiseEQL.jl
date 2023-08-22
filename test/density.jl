using ..StepwiseEQL
using EpithelialDynamics1D
using DataInterpolations
using OrdinaryDiffEq
using LinearSolve
using Random
using CairoMakie
using StatsBase
using ReferenceTests
using MovingBoundaryProblems1D
const EQL = StepwiseEQL

@testset "Derivatives" begin
    for _ in 1:1000
        a, b, c = 15rand(3)
        f = x -> a * x^2 + b * x + c
        f′ = x -> 2 * a * x + b
        f′′ = x -> 2 * a
        x1 = 5rand()
        x2 = x1 + 2rand()
        x3 = x2 + 3rand()

        @test EQL.forward_dfdx(f(x1), f(x2), f(x3), x1, x2, x3) ≈ f′(x1)
        @test EQL.central_dfdx(f(x1), f(x2), f(x3), x1, x2, x3) ≈ f′(x2)
        @test EQL.backward_dfdx(f(x1), f(x2), f(x3), x1, x2, x3) ≈ f′(x3)
        @test EQL.forward_d²fdx²(f(x1), f(x2), f(x3), x1, x2, x3) ≈ f′′(x1)
        @test EQL.central_d²fdx²(f(x1), f(x2), f(x3), x1, x2, x3) ≈ f′′(x2)
        @test EQL.backward_d²fdx²(f(x1), f(x2), f(x3), x1, x2, x3) ≈ f′′(x3)
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

    for j in 2:length(sol)
        qʲ = node_densities(sol.u[j])
        for i in 1:length(sol.u[j])
            # q 
            q1ᵢʲ = EQL.cell_density(sol, i, j)
            q2ᵢʲ = EQL.cell_density(sol.u[j], i)
            q3ᵢʲ = qʲ[i]
            q4ᵢʲ = if i == 1
                #inv(sol.u[j][i+1] - sol.u[j][i])
                LinearInterpolation([inv(sol.u[j][i+1] - sol.u[j][i]), 2inv(sol.u[j][i+2] - sol.u[j][i])], [(sol.u[j][i] + sol.u[j][i+1]) / 2, sol.u[j][i+1]])(sol.u[j][i])
            elseif i == length(sol.u[j])
                #inv(sol.u[j][end] - sol.u[j][end-1])
                LinearInterpolation([2inv(sol.u[j][i] - sol.u[j][i-2]), inv(sol.u[j][i] - sol.u[j][i-1])], [sol.u[j][i-1], (sol.u[j][i] + sol.u[j][i-1]) / 2])(sol.u[j][i])
            else
                2inv(sol.u[j][i+1] - sol.u[j][i-1])
            end
            @test q1ᵢʲ ≈ q2ᵢʲ ≈ q3ᵢʲ ≈ q4ᵢʲ

            # ∂q∂x
            ∂q1ᵢʲ = EQL.cell_∂q∂x(sol, i, j)
            ∂q2ᵢʲ = if i == 1
                q₁ = LinearInterpolation([inv(sol.u[j][i+1] - sol.u[j][i]), 2inv(sol.u[j][i+2] - sol.u[j][i])], [(sol.u[j][i] + sol.u[j][i+1]) / 2, sol.u[j][i+1]])(sol.u[j][i])#inv(sol.u[j][i+1] - sol.u[j][i])
                q₂ = 2inv(sol.u[j][i+2] - sol.u[j][i])
                q₃ = 2inv(sol.u[j][i+3] - sol.u[j][i+1])
                x₁ = sol.u[j][i]
                x₂ = sol.u[j][i+1]
                x₃ = sol.u[j][i+2]
                (q₂ - q₁) / (x₂ - x₁)
                # EQL.forward_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            elseif i == length(sol.u[j])
                q₃ = LinearInterpolation([2inv(sol.u[j][i] - sol.u[j][i-2]), inv(sol.u[j][i] - sol.u[j][i-1])], [sol.u[j][i-1], (sol.u[j][i] + sol.u[j][i-1]) / 2])(sol.u[j][i])#inv(sol.u[j][end] - sol.u[j][end-1])
                q₂ = 2inv(sol.u[j][end] - sol.u[j][end-2])
                q₁ = 2inv(sol.u[j][end-1] - sol.u[j][end-3])
                x₃ = sol.u[j][end]
                x₂ = sol.u[j][end-1]
                x₁ = sol.u[j][end-2]
                (q₃ - q₂) / (x₃ - x₂)
                # EQL.backward_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            elseif i == 2
                q₁ = LinearInterpolation([inv(sol.u[j][i] - sol.u[j][i-1]), 2inv(sol.u[j][i+1] - sol.u[j][i-1])], [(sol.u[j][i-1] + sol.u[j][i]) / 2, sol.u[j][i]])(sol.u[j][i-1])
                q₂ = 2inv(sol.u[j][i+1] - sol.u[j][i-1])
                q₃ = 2inv(sol.u[j][i+2] - sol.u[j][i])
                x₁ = sol.u[j][i-1]
                x₂ = sol.u[j][i]
                x₃ = sol.u[j][i+1]
                EQL.central_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            elseif i == length(sol.u[j]) - 1
                q₃ = LinearInterpolation([2inv(sol.u[j][i+1] - sol.u[j][i-1]), inv(sol.u[j][i+1] - sol.u[j][i])], [sol.u[j][i], (sol.u[j][i+1] + sol.u[j][i]) / 2])(sol.u[j][i+1])
                q₂ = 2inv(sol.u[j][end] - sol.u[j][end-2])
                q₁ = 2inv(sol.u[j][end-1] - sol.u[j][end-3])
                x₃ = sol.u[j][end]
                x₂ = sol.u[j][end-1]
                x₁ = sol.u[j][end-2]
                EQL.central_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            else
                q₁ = 2inv(sol.u[j][i] - sol.u[j][i-2])
                q₂ = 2inv(sol.u[j][i+1] - sol.u[j][i-1])
                q₃ = 2inv(sol.u[j][i+2] - sol.u[j][i])
                x₁ = sol.u[j][i-1]
                x₂ = sol.u[j][i]
                x₃ = sol.u[j][i+1]
                EQL.central_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            end
            @test ∂q1ᵢʲ ≈ ∂q2ᵢʲ

            # ∂²q∂x²
            ∂²q1ᵢʲ = EQL.cell_∂²q∂x²(sol, i, j)
            ∂²q2ᵢʲ = if i == 1
                qᵢʲ = LinearInterpolation([inv(sol.u[j][i+1] - sol.u[j][i]), 2inv(sol.u[j][i+2] - sol.u[j][i])], [(sol.u[j][i] + sol.u[j][i+1]) / 2, sol.u[j][i+1]])(sol.u[j][i])#inv(sol.u[j][i+1] - sol.u[j][i])
                qᵢ₊₁ʲ = 2inv(sol.u[j][i+2] - sol.u[j][i])
                qᵢ₊₂ʲ = 2inv(sol.u[j][i+3] - sol.u[j][i+1])
                EQL.forward_d²fdx²(qᵢʲ, qᵢ₊₁ʲ, qᵢ₊₂ʲ, sol.u[j][i], sol.u[j][i+1], sol.u[j][i+2])
            elseif i == length(sol.u[j])
                qᵢ₋₂ʲ = 2inv(sol.u[j][i-1] - sol.u[j][i-3])
                qᵢ₋₁ʲ = 2inv(sol.u[j][i] - sol.u[j][i-2])
                qᵢʲ = LinearInterpolation([2inv(sol.u[j][i] - sol.u[j][i-2]), inv(sol.u[j][i] - sol.u[j][i-1])], [sol.u[j][i-1], (sol.u[j][i] + sol.u[j][i-1]) / 2])(sol.u[j][i])
                EQL.backward_d²fdx²(qᵢ₋₂ʲ, qᵢ₋₁ʲ, qᵢʲ, sol.u[j][end-2], sol.u[j][end-1], sol.u[j][end])
            elseif i == 2
                qᵢ₋₁ʲ = LinearInterpolation([inv(sol.u[j][i] - sol.u[j][i-1]), 2inv(sol.u[j][i+1] - sol.u[j][i-1])], [(sol.u[j][i-1] + sol.u[j][i]) / 2, sol.u[j][i]])(sol.u[j][i-1])
                qᵢʲ = 2inv(sol.u[j][i+1] - sol.u[j][i-1])
                qᵢ₊₁ʲ = 2inv(sol.u[j][i+2] - sol.u[j][i])
                EQL.central_d²fdx²(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, sol.u[j][i-1], sol.u[j][i], sol.u[j][i+1])
            elseif i == length(sol.u[j]) - 1
                qᵢ₋₁ʲ = 2inv(sol.u[j][i] - sol.u[j][i-2])
                qᵢʲ = 2inv(sol.u[j][i+1] - sol.u[j][i-1])
                qᵢ₊₁ʲ = LinearInterpolation([2inv(sol.u[j][i+1] - sol.u[j][i-1]), inv(sol.u[j][i+1] - sol.u[j][i])], [sol.u[j][i], (sol.u[j][i+1] + sol.u[j][i]) / 2])(sol.u[j][i+1])
                EQL.central_d²fdx²(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, sol.u[j][end-2], sol.u[j][end-1], sol.u[j][end])
            else
                qᵢ₋₁ʲ = 2inv(sol.u[j][i] - sol.u[j][i-2])
                qᵢʲ = 2inv(sol.u[j][i+1] - sol.u[j][i-1])
                qᵢ₊₁ʲ = 2inv(sol.u[j][i+2] - sol.u[j][i])
                EQL.central_d²fdx²(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, sol.u[j][i-1], sol.u[j][i], sol.u[j][i+1])
            end
            @test ∂²q1ᵢʲ ≈ ∂²q2ᵢʲ

            # ∂q∂t 
            ∂q1ᵢʲ = EQL.cell_∂q∂t(sol, i, j)
            ∂q2ᵢʲ = if j == length(sol)
                qᵢʲ⁻² = LinearInterpolation(node_densities(sol.u[j-2]), sol.u[j-2])(sol.u[j][i])
                qᵢʲ⁻¹ = LinearInterpolation(node_densities(sol.u[j-1]), sol.u[j-1])(sol.u[j][i])
                qᵢʲ = EQL.cell_density(sol, i, j)
                EQL.backward_dfdx(qᵢʲ⁻², qᵢʲ⁻¹, qᵢʲ, sol.t[j-2], sol.t[j-1], sol.t[j])
            else
                qᵢʲ⁻¹ = LinearInterpolation(node_densities(sol.u[j-1]), sol.u[j-1])(sol.u[j][i])
                qᵢʲ = EQL.cell_density(sol, i, j)
                qᵢʲ⁺¹ = LinearInterpolation(node_densities(sol.u[j+1]), sol.u[j+1])(sol.u[j][i])
                EQL.central_dfdx(qᵢʲ⁻¹, qᵢʲ, qᵢʲ⁺¹, sol.t[j-1], sol.t[j], sol.t[j+1])
            end
            @test ∂q1ᵢʲ ≈ ∂q2ᵢʲ
        end
    end
end

@testset "AveragedODESolution" begin
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
    esol = solve(ens_prob, Tsit5(); trajectories=50, saveat=0.1)

    rnd = rand(1:50, 20)
    (; q, r, means, lowers, uppers, knots) = node_densities(esol; num_knots=400, indices=rnd)
    asol = EQL.AveragedODESolution(esol, 400, rnd)
    @test length(asol) == length(esol[1].t)
    @test asol.u == knots
    @test length(asol.u[1]) == 400
    @test asol.t == esol[1].t
    @test asol.q == means
    @test asol.cell_sol === esol

    for j in 2:length(esol[1].t)
        for i in 1:400
            @test EQL.cell_density(asol, i, j) == asol.q[j][i] == means[j][i]
            @inferred EQL.cell_density(asol, i, j)

            # ∂q∂x
            ∂q1ᵢʲ = EQL.cell_∂q∂x(asol, i, j)
            @inferred EQL.cell_∂q∂x(asol, i, j)
            ∂q2ᵢʲ = if i == 1
                q₁ = means[j][i]
                q₂ = means[j][i+1]
                q₃ = means[j][i+2]
                x₁ = knots[j][i]
                x₂ = knots[j][i+1]
                x₃ = knots[j][i+2]
                (q₂ - q₁) / (x₂ - x₁)
                #EQL.forward_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            elseif i == length(asol.u[j])
                q₁ = means[j][end-2]
                q₂ = means[j][end-1]
                q₃ = means[j][end]
                x₁ = knots[j][end-2]
                x₂ = knots[j][end-1]
                x₃ = knots[j][end]
                (q₃ - q₂) / (x₃ - x₂)
                #EQL.backward_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            elseif i == 2
                q₁ = means[j][i-1]
                q₂ = means[j][i]
                q₃ = means[j][i+1]
                x₁ = knots[j][i-1]
                x₂ = knots[j][i]
                x₃ = knots[j][i+1]
                EQL.central_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            elseif i == length(asol.u[j]) - 1
                q₁ = means[j][end-2]
                q₂ = means[j][end-1]
                q₃ = means[j][end]
                x₁ = knots[j][end-2]
                x₂ = knots[j][end-1]
                x₃ = knots[j][end]
                EQL.central_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            else
                q₁ = means[j][i-1]
                q₂ = means[j][i]
                q₃ = means[j][i+1]
                x₁ = knots[j][i-1]
                x₂ = knots[j][i]
                x₃ = knots[j][i+1]
                EQL.central_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            end
            @test ∂q1ᵢʲ ≈ ∂q2ᵢʲ

            # ∂²q∂x²
            ∂²q1ᵢʲ = EQL.cell_∂²q∂x²(asol, i, j)
            @inferred EQL.cell_∂²q∂x²(asol, i, j)
            ∂²q2ᵢʲ = if i == 1
                qᵢʲ = means[j][i]
                qᵢ₊₁ʲ = means[j][i+1]
                qᵢ₊₂ʲ = means[j][i+2]
                xᵢʲ = knots[j][i]
                xᵢ₊₁ʲ = knots[j][i+1]
                xᵢ₊₂ʲ = knots[j][i+2]
                EQL.forward_d²fdx²(qᵢʲ, qᵢ₊₁ʲ, qᵢ₊₂ʲ, xᵢʲ, xᵢ₊₁ʲ, xᵢ₊₂ʲ)
            elseif i == length(asol.u[j])
                qᵢ₋₂ʲ = means[j][end-2]
                qᵢ₋₁ʲ = means[j][end-1]
                qᵢʲ = means[j][end]
                xᵢ₋₂ʲ = knots[j][end-2]
                xᵢ₋₁ʲ = knots[j][end-1]
                xᵢʲ = knots[j][end]
                EQL.backward_d²fdx²(qᵢ₋₂ʲ, qᵢ₋₁ʲ, qᵢʲ, xᵢ₋₂ʲ, xᵢ₋₁ʲ, xᵢʲ)
            elseif i == 2
                qᵢ₋₁ʲ = means[j][i-1]
                qᵢʲ = means[j][i]
                qᵢ₊₁ʲ = means[j][i+1]
                xᵢ₋₁ʲ = knots[j][i-1]
                xᵢʲ = knots[j][i]
                xᵢ₊₁ʲ = knots[j][i+1]
                EQL.central_d²fdx²(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, xᵢ₋₁ʲ, xᵢʲ, xᵢ₊₁ʲ)
            elseif i == length(asol.u[j]) - 1
                qᵢ₋₁ʲ = means[j][i-1]
                qᵢʲ = means[j][i]
                qᵢ₊₁ʲ = means[j][i+1]
                xᵢ₋₁ʲ = knots[j][i-1]
                xᵢʲ = knots[j][i]
                xᵢ₊₁ʲ = knots[j][i+1]
                EQL.central_d²fdx²(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, xᵢ₋₁ʲ, xᵢʲ, xᵢ₊₁ʲ)
            else
                qᵢ₋₁ʲ = means[j][i-1]
                qᵢʲ = means[j][i]
                qᵢ₊₁ʲ = means[j][i+1]
                xᵢ₋₁ʲ = knots[j][i-1]
                xᵢʲ = knots[j][i]
                xᵢ₊₁ʲ = knots[j][i+1]
                EQL.central_d²fdx²(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, xᵢ₋₁ʲ, xᵢʲ, xᵢ₊₁ʲ)
            end
            @test ∂²q1ᵢʲ ≈ ∂²q2ᵢʲ

            # ∂q∂t 
            ∂q1ᵢʲ = EQL.cell_∂q∂t(asol, i, j)
            @inferred EQL.cell_∂q∂t(asol, i, j)
            ∂q2ᵢʲ = if j == length(asol)
                qᵢʲ⁻² = LinearInterpolation(means[j-2], knots[j-2])(knots[j][i])
                qᵢʲ⁻¹ = LinearInterpolation(means[j-1], knots[j-1])(knots[j][i])
                qᵢʲ = means[j][i]
                EQL.backward_dfdx(qᵢʲ⁻², qᵢʲ⁻¹, qᵢʲ, asol.t[j-2], asol.t[j-1], asol.t[j])
            else
                qᵢʲ⁻¹ = LinearInterpolation(means[j-1], knots[j-1])(knots[j][i])
                qᵢʲ = means[j][i]
                qᵢʲ⁺¹ = LinearInterpolation(means[j+1], knots[j+1])(knots[j][i])
                EQL.central_dfdx(qᵢʲ⁻¹, qᵢʲ, qᵢʲ⁺¹, asol.t[j-1], asol.t[j], asol.t[j+1])
            end
            @test ∂q1ᵢʲ ≈ ∂q2ᵢʲ
        end
    end
end

@testset "MovingBoundaryProblem with Proliferation" begin
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
        proliferation_law_parameters=Gp,
        fix_right=false)
    ens_prob = EnsembleProblem(prob)
    esol = solve(ens_prob, Tsit5(), EnsembleThreads(), trajectories=100, saveat=0.1)

    for k in [rand(1:length(esol), 25); 1; length(esol)]
        sol = esol[k]
        for j in [rand(2:length(sol), 50); 2; length(sol); length(sol) - 1; length(sol) - 2]
            qʲ = node_densities(sol.u[j])
            for i in 1:length(sol.u[j])
                # q 
                q1ᵢʲ = EQL.cell_density(sol, i, j)
                q2ᵢʲ = EQL.cell_density(sol.u[j], i)
                q3ᵢʲ = qʲ[i]
                q4ᵢʲ = if i == 1
                    LinearInterpolation([inv(sol.u[j][i+1] - sol.u[j][i]), 2inv(sol.u[j][i+2] - sol.u[j][i])], [(sol.u[j][i] + sol.u[j][i+1]) / 2, sol.u[j][i+1]])(sol.u[j][i])
                elseif i == length(sol.u[j])
                    LinearInterpolation([2inv(sol.u[j][i] - sol.u[j][i-2]), inv(sol.u[j][i] - sol.u[j][i-1])], [sol.u[j][i-1], (sol.u[j][i] + sol.u[j][i-1]) / 2])(sol.u[j][i])
                else
                    2inv(sol.u[j][i+1] - sol.u[j][i-1])
                end
                @test q1ᵢʲ ≈ q2ᵢʲ ≈ q3ᵢʲ ≈ q4ᵢʲ

                # ∂q∂x
                ∂q1ᵢʲ = EQL.cell_∂q∂x(sol, i, j)
                ∂q2ᵢʲ = if i == 1
                    q₁ = LinearInterpolation([inv(sol.u[j][i+1] - sol.u[j][i]), 2inv(sol.u[j][i+2] - sol.u[j][i])], [(sol.u[j][i] + sol.u[j][i+1]) / 2, sol.u[j][i+1]])(sol.u[j][i])#inv(sol.u[j][i+1] - sol.u[j][i])
                    q₂ = 2inv(sol.u[j][i+2] - sol.u[j][i])
                    q₃ = 2inv(sol.u[j][i+3] - sol.u[j][i+1])
                    x₁ = sol.u[j][i]
                    x₂ = sol.u[j][i+1]
                    x₃ = sol.u[j][i+2]
                    (q₂ - q₁) / (x₂ - x₁)
                    #EQL.forward_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
                elseif i == length(sol.u[j])
                    q₃ = EQL.cell_density(sol, i, j)
                    q₂ = 2inv(sol.u[j][end] - sol.u[j][end-2])
                    q₁ = 2inv(sol.u[j][end-1] - sol.u[j][end-3])
                    x₃ = sol.u[j][end]
                    x₂ = sol.u[j][end-1]
                    x₁ = sol.u[j][end-2]
                    (q₃ - q₂) / (x₃ - x₂)
                    #EQL.backward_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
                elseif i == 2
                    q₁ = EQL.cell_density(sol, i - 1, j)
                    q₂ = 2inv(sol.u[j][i+1] - sol.u[j][i-1])
                    q₃ = 2inv(sol.u[j][i+2] - sol.u[j][i])
                    x₁ = sol.u[j][i-1]
                    x₂ = sol.u[j][i]
                    x₃ = sol.u[j][i+1]
                    EQL.central_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
                elseif i == length(sol.u[j]) - 1
                    q₃ = EQL.cell_density(sol, i + 1, j)
                    q₂ = 2inv(sol.u[j][end] - sol.u[j][end-2])
                    q₁ = 2inv(sol.u[j][end-1] - sol.u[j][end-3])
                    x₃ = sol.u[j][end]
                    x₂ = sol.u[j][end-1]
                    x₁ = sol.u[j][end-2]
                    EQL.central_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
                else
                    q₁ = 2inv(sol.u[j][i] - sol.u[j][i-2])
                    q₂ = 2inv(sol.u[j][i+1] - sol.u[j][i-1])
                    q₃ = 2inv(sol.u[j][i+2] - sol.u[j][i])
                    x₁ = sol.u[j][i-1]
                    x₂ = sol.u[j][i]
                    x₃ = sol.u[j][i+1]
                    EQL.central_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
                end
                @test ∂q1ᵢʲ ≈ ∂q2ᵢʲ

                # ∂²q∂x²
                ∂²q1ᵢʲ = EQL.cell_∂²q∂x²(sol, i, j)
                ∂²q2ᵢʲ = if i == 1
                    qᵢʲ = EQL.cell_density(sol, i, j)
                    qᵢ₊₁ʲ = 2inv(sol.u[j][i+2] - sol.u[j][i])
                    qᵢ₊₂ʲ = 2inv(sol.u[j][i+3] - sol.u[j][i+1])
                    EQL.forward_d²fdx²(qᵢʲ, qᵢ₊₁ʲ, qᵢ₊₂ʲ, sol.u[j][i], sol.u[j][i+1], sol.u[j][i+2])
                elseif i == length(sol.u[j])
                    qᵢ₋₂ʲ = 2inv(sol.u[j][i-1] - sol.u[j][i-3])
                    qᵢ₋₁ʲ = 2inv(sol.u[j][i] - sol.u[j][i-2])
                    qᵢʲ = EQL.cell_density(sol, i, j)
                    EQL.backward_d²fdx²(qᵢ₋₂ʲ, qᵢ₋₁ʲ, qᵢʲ, sol.u[j][end-2], sol.u[j][end-1], sol.u[j][end])
                elseif i == 2
                    qᵢ₋₁ʲ = EQL.cell_density(sol, i - 1, j)
                    qᵢʲ = 2inv(sol.u[j][i+1] - sol.u[j][i-1])
                    qᵢ₊₁ʲ = 2inv(sol.u[j][i+2] - sol.u[j][i])
                    EQL.central_d²fdx²(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, sol.u[j][i-1], sol.u[j][i], sol.u[j][i+1])
                elseif i == length(sol.u[j]) - 1
                    qᵢ₋₁ʲ = 2inv(sol.u[j][i] - sol.u[j][i-2])
                    qᵢʲ = 2inv(sol.u[j][i+1] - sol.u[j][i-1])
                    qᵢ₊₁ʲ = EQL.cell_density(sol, i + 1, j)
                    EQL.central_d²fdx²(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, sol.u[j][end-2], sol.u[j][end-1], sol.u[j][end])
                else
                    qᵢ₋₁ʲ = 2inv(sol.u[j][i] - sol.u[j][i-2])
                    qᵢʲ = 2inv(sol.u[j][i+1] - sol.u[j][i-1])
                    qᵢ₊₁ʲ = 2inv(sol.u[j][i+2] - sol.u[j][i])
                    EQL.central_d²fdx²(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, sol.u[j][i-1], sol.u[j][i], sol.u[j][i+1])
                end
                @test ∂²q1ᵢʲ ≈ ∂²q2ᵢʲ

                # ∂q∂t 
                ∂q1ᵢʲ = EQL.cell_∂q∂t(sol, i, j)
                ∂q2ᵢʲ = if j == length(sol)
                    qᵢʲ⁻² = LinearInterpolation(node_densities(sol.u[j-2]), sol.u[j-2])(sol.u[j][i])
                    qᵢʲ⁻¹ = LinearInterpolation(node_densities(sol.u[j-1]), sol.u[j-1])(sol.u[j][i])
                    qᵢʲ = EQL.cell_density(sol, i, j)
                    EQL.backward_dfdx(qᵢʲ⁻², qᵢʲ⁻¹, qᵢʲ, sol.t[j-2], sol.t[j-1], sol.t[j])
                else
                    qᵢʲ⁻¹ = LinearInterpolation(node_densities(sol.u[j-1]), sol.u[j-1])(sol.u[j][i])
                    qᵢʲ = EQL.cell_density(sol, i, j)
                    qᵢʲ⁺¹ = LinearInterpolation(node_densities(sol.u[j+1]), sol.u[j+1])(sol.u[j][i])
                    EQL.central_dfdx(qᵢʲ⁻¹, qᵢʲ, qᵢʲ⁺¹, sol.t[j-1], sol.t[j], sol.t[j+1])
                end
                @test ∂q1ᵢʲ ≈ ∂q2ᵢʲ
            end
            # dLdt 
            dLdt1 = EQL.cell_dLdt(sol, j)
            @inferred EQL.cell_dLdt(sol, j)
            dLdt2 = if j == length(sol)
                L3 = sol.u[j][end]
                L2 = sol.u[j-1][end]
                L1 = sol.u[j-2][end]
                t3 = sol.t[j]
                t2 = sol.t[j-1]
                t1 = sol.t[j-2]
                EQL.backward_dfdx(L1, L2, L3, t1, t2, t3)
            else
                L1 = sol.u[j-1][end]
                L2 = sol.u[j][end]
                L3 = sol.u[j+1][end]
                t1 = sol.t[j-1]
                t2 = sol.t[j]
                t3 = sol.t[j+1]
                EQL.central_dfdx(L1, L2, L3, t1, t2, t3)
            end
            @test dLdt1 ≈ dLdt2
        end
    end
end

@testset "MovingBoundaryProblem with Proliferation with AveragedODESolution" begin
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
        proliferation_law_parameters=Gp,
        fix_right=false)
    ens_prob = EnsembleProblem(prob)
    esol = solve(ens_prob, Tsit5(), EnsembleThreads(), trajectories=100, saveat=0.1)

    # stat = (minimum, maximum)
    rnd = [rand(eachindex(esol), 20); 1; 100]
    (; q, r, means, lowers, uppers, knots) = node_densities(esol; num_knots=400, indices=rnd, stat=(minimum, maximum),extrapolate=true)
    Lstats = leading_edges(esol; indices=rnd)
    for i in eachindex(knots)
        @test knots[i] ≈ LinRange(minimum(first.(getindex.(r, i))), maximum(last.(getindex.(r, i))), 400)
    end
    asol = EQL.AveragedODESolution(esol, 400, rnd, LinearInterpolation{true}, (minimum, maximum))
    @test length(asol) == length(esol[1].t)
    @test asol.u == knots
    @test length(asol.u[1]) == 400
    @test asol.t == esol[1].t
    @test asol.q == means
    @test asol.cell_sol === esol
    for j in 2:length(esol[1].t)
        for i in 1:400
            @test EQL.cell_density(asol, i, j) == asol.q[j][i] == means[j][i]
            @inferred EQL.cell_density(asol, i, j)

            # ∂q∂x
            ∂q1ᵢʲ = EQL.cell_∂q∂x(asol, i, j)
            @inferred EQL.cell_∂q∂x(asol, i, j)
            ∂q2ᵢʲ = if i == 1
                q₁ = means[j][i]
                q₂ = means[j][i+1]
                q₃ = means[j][i+2]
                x₁ = knots[j][i]
                x₂ = knots[j][i+1]
                x₃ = knots[j][i+2]
                (q₂ - q₁)/(x₂ - x₁)
                #EQL.forward_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            elseif i == length(asol.u[j])
                q₁ = means[j][end-2]
                q₂ = means[j][end-1]
                q₃ = means[j][end]
                x₁ = knots[j][end-2]
                x₂ = knots[j][end-1]
                x₃ = knots[j][end]
                (q₃ - q₂)/(x₃ - x₂)
                #EQL.backward_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            elseif i == 2
                q₁ = means[j][i-1]
                q₂ = means[j][i]
                q₃ = means[j][i+1]
                x₁ = knots[j][i-1]
                x₂ = knots[j][i]
                x₃ = knots[j][i+1]
                EQL.central_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            elseif i == length(asol.u[j]) - 1
                q₁ = means[j][end-2]
                q₂ = means[j][end-1]
                q₃ = means[j][end]
                x₁ = knots[j][end-2]
                x₂ = knots[j][end-1]
                x₃ = knots[j][end]
                EQL.central_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            else
                q₁ = means[j][i-1]
                q₂ = means[j][i]
                q₃ = means[j][i+1]
                x₁ = knots[j][i-1]
                x₂ = knots[j][i]
                x₃ = knots[j][i+1]
                EQL.central_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            end
            @test ∂q1ᵢʲ ≈ ∂q2ᵢʲ

            # ∂²q∂x²
            ∂²q1ᵢʲ = EQL.cell_∂²q∂x²(asol, i, j)
            @inferred EQL.cell_∂²q∂x²(asol, i, j)
            ∂²q2ᵢʲ = if i == 1
                qᵢʲ = means[j][i]
                qᵢ₊₁ʲ = means[j][i+1]
                qᵢ₊₂ʲ = means[j][i+2]
                xᵢʲ = knots[j][i]
                xᵢ₊₁ʲ = knots[j][i+1]
                xᵢ₊₂ʲ = knots[j][i+2]
                EQL.forward_d²fdx²(qᵢʲ, qᵢ₊₁ʲ, qᵢ₊₂ʲ, xᵢʲ, xᵢ₊₁ʲ, xᵢ₊₂ʲ)
            elseif i == length(asol.u[j])
                qᵢ₋₂ʲ = means[j][end-2]
                qᵢ₋₁ʲ = means[j][end-1]
                qᵢʲ = means[j][end]
                xᵢ₋₂ʲ = knots[j][end-2]
                xᵢ₋₁ʲ = knots[j][end-1]
                xᵢʲ = knots[j][end]
                EQL.backward_d²fdx²(qᵢ₋₂ʲ, qᵢ₋₁ʲ, qᵢʲ, xᵢ₋₂ʲ, xᵢ₋₁ʲ, xᵢʲ)
            elseif i == 2
                qᵢ₋₁ʲ = means[j][i-1]
                qᵢʲ = means[j][i]
                qᵢ₊₁ʲ = means[j][i+1]
                xᵢ₋₁ʲ = knots[j][i-1]
                xᵢʲ = knots[j][i]
                xᵢ₊₁ʲ = knots[j][i+1]
                EQL.central_d²fdx²(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, xᵢ₋₁ʲ, xᵢʲ, xᵢ₊₁ʲ)
            elseif i == length(asol.u[j]) - 1
                qᵢ₋₁ʲ = means[j][i-1]
                qᵢʲ = means[j][i]
                qᵢ₊₁ʲ = means[j][i+1]
                xᵢ₋₁ʲ = knots[j][i-1]
                xᵢʲ = knots[j][i]
                xᵢ₊₁ʲ = knots[j][i+1]
                EQL.central_d²fdx²(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, xᵢ₋₁ʲ, xᵢʲ, xᵢ₊₁ʲ)
            else
                qᵢ₋₁ʲ = means[j][i-1]
                qᵢʲ = means[j][i]
                qᵢ₊₁ʲ = means[j][i+1]
                xᵢ₋₁ʲ = knots[j][i-1]
                xᵢʲ = knots[j][i]
                xᵢ₊₁ʲ = knots[j][i+1]
                EQL.central_d²fdx²(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, xᵢ₋₁ʲ, xᵢʲ, xᵢ₊₁ʲ)
            end
            @test ∂²q1ᵢʲ ≈ ∂²q2ᵢʲ

            # ∂q∂t 
            ∂q1ᵢʲ = EQL.cell_∂q∂t(asol, i, j)
            @inferred EQL.cell_∂q∂t(asol, i, j)
            ∂q2ᵢʲ = if j == length(asol)
                qᵢʲ⁻² = LinearInterpolation(means[j-2], knots[j-2])(knots[j][i])
                qᵢʲ⁻¹ = LinearInterpolation(means[j-1], knots[j-1])(knots[j][i])
                qᵢʲ = means[j][i]
                EQL.backward_dfdx(qᵢʲ⁻², qᵢʲ⁻¹, qᵢʲ, asol.t[j-2], asol.t[j-1], asol.t[j])
            else
                qᵢʲ⁻¹ = LinearInterpolation(means[j-1], knots[j-1])(knots[j][i])
                qᵢʲ = means[j][i]
                qᵢʲ⁺¹ = LinearInterpolation(means[j+1], knots[j+1])(knots[j][i])
                EQL.central_dfdx(qᵢʲ⁻¹, qᵢʲ, qᵢʲ⁺¹, asol.t[j-1], asol.t[j], asol.t[j+1])
            end
            @test ∂q1ᵢʲ ≈ ∂q2ᵢʲ

        end
        # dLdt 
        dLdt1 = EQL.cell_dLdt(asol, j)
        @inferred EQL.cell_dLdt(asol, j)
        dLdt2 = if j == length(asol)
            L3 = asol.u[j][end]
            L2 = asol.u[j-1][end]
            L1 = asol.u[j-2][end]
            t3 = asol.t[j]
            t2 = asol.t[j-1]
            t1 = asol.t[j-2]
            EQL.backward_dfdx(L1, L2, L3, t1, t2, t3)
        else
            L1 = asol.u[j-1][end]
            L2 = asol.u[j][end]
            L3 = asol.u[j+1][end]
            t1 = asol.t[j-1]
            t2 = asol.t[j]
            t3 = asol.t[j+1]
            EQL.central_dfdx(L1, L2, L3, t1, t2, t3)
        end
        @test dLdt1 ≈ dLdt2
    end

    # stat = mean
    rnd = [rand(eachindex(esol), 20); 1; 100]
    (; q, r, means, lowers, uppers, knots) = node_densities(esol; num_knots=400, indices=rnd, stat=mean,extrapolate=true)
    Lstats = leading_edges(esol; indices=rnd)
    for i in eachindex(knots)
        @test knots[i] ≈ LinRange(mean(first.(getindex.(r, i))), mean(last.(getindex.(r, i))), 400)
    end
    asol = EQL.AveragedODESolution(esol, 400, rnd, LinearInterpolation{true}, mean)
    @test length(asol) == length(esol[1].t)
    @test asol.u == knots
    @test length(asol.u[1]) == 400
    @test asol.t == esol[1].t
    @test asol.q == means
    @test asol.cell_sol === esol
    for j in 2:length(esol[1].t)
        for i in 1:400
            @test EQL.cell_density(asol, i, j) == asol.q[j][i] == means[j][i]
            @inferred EQL.cell_density(asol, i, j)

            # ∂q∂x
            ∂q1ᵢʲ = EQL.cell_∂q∂x(asol, i, j)
            @inferred EQL.cell_∂q∂x(asol, i, j)
            ∂q2ᵢʲ = if i == 1
                q₁ = means[j][i]
                q₂ = means[j][i+1]
                q₃ = means[j][i+2]
                x₁ = knots[j][i]
                x₂ = knots[j][i+1]
                x₃ = knots[j][i+2]
                (q₂ - q₁)/(x₂ - x₁)
                #EQL.forward_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            elseif i == length(asol.u[j])
                q₁ = means[j][end-2]
                q₂ = means[j][end-1]
                q₃ = means[j][end]
                x₁ = knots[j][end-2]
                x₂ = knots[j][end-1]
                x₃ = knots[j][end]
                (q₃ - q₂)/(x₃ - x₂)
                #EQL.backward_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            elseif i == 2
                q₁ = means[j][i-1]
                q₂ = means[j][i]
                q₃ = means[j][i+1]
                x₁ = knots[j][i-1]
                x₂ = knots[j][i]
                x₃ = knots[j][i+1]
                EQL.central_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            elseif i == length(asol.u[j]) - 1
                q₁ = means[j][end-2]
                q₂ = means[j][end-1]
                q₃ = means[j][end]
                x₁ = knots[j][end-2]
                x₂ = knots[j][end-1]
                x₃ = knots[j][end]
                EQL.central_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            else
                q₁ = means[j][i-1]
                q₂ = means[j][i]
                q₃ = means[j][i+1]
                x₁ = knots[j][i-1]
                x₂ = knots[j][i]
                x₃ = knots[j][i+1]
                EQL.central_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            end
            @test ∂q1ᵢʲ ≈ ∂q2ᵢʲ

            # ∂²q∂x²
            ∂²q1ᵢʲ = EQL.cell_∂²q∂x²(asol, i, j)
            @inferred EQL.cell_∂²q∂x²(asol, i, j)
            ∂²q2ᵢʲ = if i == 1
                qᵢʲ = means[j][i]
                qᵢ₊₁ʲ = means[j][i+1]
                qᵢ₊₂ʲ = means[j][i+2]
                xᵢʲ = knots[j][i]
                xᵢ₊₁ʲ = knots[j][i+1]
                xᵢ₊₂ʲ = knots[j][i+2]
                EQL.forward_d²fdx²(qᵢʲ, qᵢ₊₁ʲ, qᵢ₊₂ʲ, xᵢʲ, xᵢ₊₁ʲ, xᵢ₊₂ʲ)
            elseif i == length(asol.u[j])
                qᵢ₋₂ʲ = means[j][end-2]
                qᵢ₋₁ʲ = means[j][end-1]
                qᵢʲ = means[j][end]
                xᵢ₋₂ʲ = knots[j][end-2]
                xᵢ₋₁ʲ = knots[j][end-1]
                xᵢʲ = knots[j][end]
                EQL.backward_d²fdx²(qᵢ₋₂ʲ, qᵢ₋₁ʲ, qᵢʲ, xᵢ₋₂ʲ, xᵢ₋₁ʲ, xᵢʲ)
            elseif i == 2
                qᵢ₋₁ʲ = means[j][i-1]
                qᵢʲ = means[j][i]
                qᵢ₊₁ʲ = means[j][i+1]
                xᵢ₋₁ʲ = knots[j][i-1]
                xᵢʲ = knots[j][i]
                xᵢ₊₁ʲ = knots[j][i+1]
                EQL.central_d²fdx²(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, xᵢ₋₁ʲ, xᵢʲ, xᵢ₊₁ʲ)
            elseif i == length(asol.u[j]) - 1
                qᵢ₋₁ʲ = means[j][i-1]
                qᵢʲ = means[j][i]
                qᵢ₊₁ʲ = means[j][i+1]
                xᵢ₋₁ʲ = knots[j][i-1]
                xᵢʲ = knots[j][i]
                xᵢ₊₁ʲ = knots[j][i+1]
                EQL.central_d²fdx²(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, xᵢ₋₁ʲ, xᵢʲ, xᵢ₊₁ʲ)
            else
                qᵢ₋₁ʲ = means[j][i-1]
                qᵢʲ = means[j][i]
                qᵢ₊₁ʲ = means[j][i+1]
                xᵢ₋₁ʲ = knots[j][i-1]
                xᵢʲ = knots[j][i]
                xᵢ₊₁ʲ = knots[j][i+1]
                EQL.central_d²fdx²(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, xᵢ₋₁ʲ, xᵢʲ, xᵢ₊₁ʲ)
            end
            @test ∂²q1ᵢʲ ≈ ∂²q2ᵢʲ

            # ∂q∂t 
            ∂q1ᵢʲ = EQL.cell_∂q∂t(asol, i, j)
            @inferred EQL.cell_∂q∂t(asol, i, j)
            ∂q2ᵢʲ = if j == length(asol)
                qᵢʲ⁻² = LinearInterpolation(means[j-2], knots[j-2])(knots[j][i])
                qᵢʲ⁻¹ = LinearInterpolation(means[j-1], knots[j-1])(knots[j][i])
                qᵢʲ = means[j][i]
                EQL.backward_dfdx(qᵢʲ⁻², qᵢʲ⁻¹, qᵢʲ, asol.t[j-2], asol.t[j-1], asol.t[j])
            else
                qᵢʲ⁻¹ = LinearInterpolation(means[j-1], knots[j-1])(knots[j][i])
                qᵢʲ = means[j][i]
                qᵢʲ⁺¹ = LinearInterpolation(means[j+1], knots[j+1])(knots[j][i])
                EQL.central_dfdx(qᵢʲ⁻¹, qᵢʲ, qᵢʲ⁺¹, asol.t[j-1], asol.t[j], asol.t[j+1])
            end
            @test ∂q1ᵢʲ ≈ ∂q2ᵢʲ

        end
        # dLdt 
        dLdt1 = EQL.cell_dLdt(asol, j)
        @inferred EQL.cell_dLdt(asol, j)
        dLdt2 = if j == length(asol)
            L3 = asol.u[j][end]
            L2 = asol.u[j-1][end]
            L1 = asol.u[j-2][end]
            t3 = asol.t[j]
            t2 = asol.t[j-1]
            t1 = asol.t[j-2]
            EQL.backward_dfdx(L1, L2, L3, t1, t2, t3)
        else
            L1 = asol.u[j-1][end]
            L2 = asol.u[j][end]
            L3 = asol.u[j+1][end]
            t1 = asol.t[j-1]
            t2 = asol.t[j]
            t3 = asol.t[j+1]
            EQL.central_dfdx(L1, L2, L3, t1, t2, t3)
        end
        @test dLdt1 ≈ dLdt2
    end
end

@testset "MovingBoundaryProblem with no Proliferation" begin
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
    prob = CellProblem(;
        final_time,
        initial_condition=cell_nodes,
        damping_constant,
        force_law,
        force_law_parameters,
        fix_right=false)
    sol = solve(prob, Tsit5(), saveat=0.1)

    for j in [rand(2:length(sol), 50); 2; length(sol); length(sol) - 1; length(sol) - 2]
        qʲ = node_densities(sol.u[j])
        for i in 1:length(sol.u[j])
            # q 
            q1ᵢʲ = EQL.cell_density(sol, i, j)
            q2ᵢʲ = EQL.cell_density(sol.u[j], i)
            q3ᵢʲ = qʲ[i]
            q4ᵢʲ = if i == 1
                LinearInterpolation([inv(sol.u[j][i+1] - sol.u[j][i]), 2inv(sol.u[j][i+2] - sol.u[j][i])], [(sol.u[j][i] + sol.u[j][i+1]) / 2, sol.u[j][i+1]])(sol.u[j][i])
            elseif i == length(sol.u[j])
                LinearInterpolation([2inv(sol.u[j][i] - sol.u[j][i-2]), inv(sol.u[j][i] - sol.u[j][i-1])], [sol.u[j][i-1], (sol.u[j][i] + sol.u[j][i-1]) / 2])(sol.u[j][i])
            else
                2inv(sol.u[j][i+1] - sol.u[j][i-1])
            end
            @test q1ᵢʲ ≈ q2ᵢʲ ≈ q3ᵢʲ ≈ q4ᵢʲ

            # ∂q∂x
            ∂q1ᵢʲ = EQL.cell_∂q∂x(sol, i, j)
            ∂q2ᵢʲ = if i == 1
                q₁ = LinearInterpolation([inv(sol.u[j][i+1] - sol.u[j][i]), 2inv(sol.u[j][i+2] - sol.u[j][i])], [(sol.u[j][i] + sol.u[j][i+1]) / 2, sol.u[j][i+1]])(sol.u[j][i])
                q₂ = 2inv(sol.u[j][i+2] - sol.u[j][i])
                q₃ = 2inv(sol.u[j][i+3] - sol.u[j][i+1])
                x₁ = sol.u[j][i]
                x₂ = sol.u[j][i+1]
                x₃ = sol.u[j][i+2]
                (q₂ - q₁)/(x₂ - x₁)
                #EQL.forward_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            elseif i == length(sol.u[j])
                q₃ = LinearInterpolation([2inv(sol.u[j][i] - sol.u[j][i-2]), inv(sol.u[j][i] - sol.u[j][i-1])], [sol.u[j][i-1], (sol.u[j][i] + sol.u[j][i-1]) / 2])(sol.u[j][i])
                q₂ = 2inv(sol.u[j][end] - sol.u[j][end-2])
                q₁ = 2inv(sol.u[j][end-1] - sol.u[j][end-3])
                x₃ = sol.u[j][end]
                x₂ = sol.u[j][end-1]
                x₁ = sol.u[j][end-2]
                (q₃ - q₂)/(x₃ - x₂)
                #EQL.backward_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            elseif i == 2
                q₁ = EQL.cell_density(sol, i - 1, j)
                q₂ = 2inv(sol.u[j][i+1] - sol.u[j][i-1])
                q₃ = 2inv(sol.u[j][i+2] - sol.u[j][i])
                x₁ = sol.u[j][i-1]
                x₂ = sol.u[j][i]
                x₃ = sol.u[j][i+1]
                EQL.central_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            elseif i == length(sol.u[j]) - 1
                q₃ = EQL.cell_density(sol, i + 1, j)
                q₂ = 2inv(sol.u[j][end] - sol.u[j][end-2])
                q₁ = 2inv(sol.u[j][end-1] - sol.u[j][end-3])
                x₃ = sol.u[j][end]
                x₂ = sol.u[j][end-1]
                x₁ = sol.u[j][end-2]
                EQL.central_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            else
                q₁ = 2inv(sol.u[j][i] - sol.u[j][i-2])
                q₂ = 2inv(sol.u[j][i+1] - sol.u[j][i-1])
                q₃ = 2inv(sol.u[j][i+2] - sol.u[j][i])
                x₁ = sol.u[j][i-1]
                x₂ = sol.u[j][i]
                x₃ = sol.u[j][i+1]
                EQL.central_dfdx(q₁, q₂, q₃, x₁, x₂, x₃)
            end
            @test ∂q1ᵢʲ ≈ ∂q2ᵢʲ

            # ∂²q∂x²
            ∂²q1ᵢʲ = EQL.cell_∂²q∂x²(sol, i, j)
            ∂²q2ᵢʲ = if i == 1
                qᵢʲ = EQL.cell_density(sol, i, j)
                qᵢ₊₁ʲ = 2inv(sol.u[j][i+2] - sol.u[j][i])
                qᵢ₊₂ʲ = 2inv(sol.u[j][i+3] - sol.u[j][i+1])
                EQL.forward_d²fdx²(qᵢʲ, qᵢ₊₁ʲ, qᵢ₊₂ʲ, sol.u[j][i], sol.u[j][i+1], sol.u[j][i+2])
            elseif i == length(sol.u[j])
                qᵢ₋₂ʲ = 2inv(sol.u[j][i-1] - sol.u[j][i-3])
                qᵢ₋₁ʲ = 2inv(sol.u[j][i] - sol.u[j][i-2])
                qᵢʲ = EQL.cell_density(sol, i, j)
                EQL.backward_d²fdx²(qᵢ₋₂ʲ, qᵢ₋₁ʲ, qᵢʲ, sol.u[j][end-2], sol.u[j][end-1], sol.u[j][end])
            elseif i == 2
                qᵢ₋₁ʲ = EQL.cell_density(sol, i - 1, j)
                qᵢʲ = 2inv(sol.u[j][i+1] - sol.u[j][i-1])
                qᵢ₊₁ʲ = 2inv(sol.u[j][i+2] - sol.u[j][i])
                EQL.central_d²fdx²(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, sol.u[j][i-1], sol.u[j][i], sol.u[j][i+1])
            elseif i == length(sol.u[j]) - 1
                qᵢ₋₁ʲ = 2inv(sol.u[j][i] - sol.u[j][i-2])
                qᵢʲ = 2inv(sol.u[j][i+1] - sol.u[j][i-1])
                qᵢ₊₁ʲ = EQL.cell_density(sol, i + 1, j)
                EQL.central_d²fdx²(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, sol.u[j][end-2], sol.u[j][end-1], sol.u[j][end])
            else
                qᵢ₋₁ʲ = 2inv(sol.u[j][i] - sol.u[j][i-2])
                qᵢʲ = 2inv(sol.u[j][i+1] - sol.u[j][i-1])
                qᵢ₊₁ʲ = 2inv(sol.u[j][i+2] - sol.u[j][i])
                EQL.central_d²fdx²(qᵢ₋₁ʲ, qᵢʲ, qᵢ₊₁ʲ, sol.u[j][i-1], sol.u[j][i], sol.u[j][i+1])
            end
            @test ∂²q1ᵢʲ ≈ ∂²q2ᵢʲ

            # ∂q∂t 
            ∂q1ᵢʲ = EQL.cell_∂q∂t(sol, i, j)
            ∂q2ᵢʲ = if j == length(sol)
                qᵢʲ⁻² = LinearInterpolation(node_densities(sol.u[j-2]), sol.u[j-2])(sol.u[j][i])
                qᵢʲ⁻¹ = LinearInterpolation(node_densities(sol.u[j-1]), sol.u[j-1])(sol.u[j][i])
                qᵢʲ = EQL.cell_density(sol, i, j)
                EQL.backward_dfdx(qᵢʲ⁻², qᵢʲ⁻¹, qᵢʲ, sol.t[j-2], sol.t[j-1], sol.t[j])
            else
                qᵢʲ⁻¹ = LinearInterpolation(node_densities(sol.u[j-1]), sol.u[j-1])(sol.u[j][i])
                qᵢʲ = EQL.cell_density(sol, i, j)
                qᵢʲ⁺¹ = LinearInterpolation(node_densities(sol.u[j+1]), sol.u[j+1])(sol.u[j][i])
                EQL.central_dfdx(qᵢʲ⁻¹, qᵢʲ, qᵢʲ⁺¹, sol.t[j-1], sol.t[j], sol.t[j+1])
            end
            @test ∂q1ᵢʲ ≈ ∂q2ᵢʲ
        end
        # dLdt 
        dLdt1 = EQL.cell_dLdt(sol, j)
        @inferred EQL.cell_dLdt(sol, j)
        dLdt2 = if j == length(sol)
            L3 = sol.u[j][end]
            L2 = sol.u[j-1][end]
            L1 = sol.u[j-2][end]
            t3 = sol.t[j]
            t2 = sol.t[j-1]
            t1 = sol.t[j-2]
            EQL.backward_dfdx(L1, L2, L3, t1, t2, t3)
        else
            L1 = sol.u[j-1][end]
            L2 = sol.u[j][end]
            L3 = sol.u[j+1][end]
            t1 = sol.t[j-1]
            t2 = sol.t[j]
            t3 = sol.t[j+1]
            EQL.central_dfdx(L1, L2, L3, t1, t2, t3)
        end
        @test dLdt1 ≈ dLdt2
    end
end

@testset "Testing derivative estimation of L using the continuum limit problem" begin
    @testset "MovingBoundaryProblem with Proliferation" begin
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
            proliferation_law_parameters=Gp,
            fix_right=false)
        ens_prob = EnsembleProblem(prob)
        Random.seed!(123)
        esol = solve(ens_prob, Tsit5(), EnsembleSerial(), trajectories=100, saveat=0.01)
        asol = EQL.AveragedODESolution(esol, 500, eachindex(esol), LinearInterpolation{true}, mean)
        asol2 = EQL.AveragedODESolution(esol, 500, eachindex(esol), LinearInterpolation{true}, (minimum, maximum))
        mb_prob = MBProblem(prob, 2500, proliferation=true)
        psol = solve(mb_prob, TRBDF2(linsolve=KLUFactorization()), saveat=esol[1].t, reltol=1e-9, abstol=1e-9)
        qstats = node_densities(esol)
        Lstats = leading_edges(esol)
        pde_L = psol[end, :]
        discrete_Lderivs = [EQL.cell_dLdt(asol, j) for j in 2:length(asol)]
        discrete_Lderivs2 = [EQL.cell_dLdt(asol2, j) for j in 2:length(asol2)]
        pde_Lderivs = [psol(psol.t[j], Val{1}; idxs=length(psol.u[j])) for j in 2:length(psol.t)]
        pde_Lderiv_fd = zeros(length(psol) - 1)
        for j in 2:length(psol.t)
            if j == length(psol.t)
                L1 = psol.u[j-2][end]
                L2 = psol.u[j-1][end]
                L3 = psol.u[j][end]
                t1 = psol.t[j-2]
                t2 = psol.t[j-1]
                t3 = psol.t[j]
                pde_Lderiv_fd[j-1] = EQL.backward_dfdx(L1, L2, L3, t1, t2, t3)
            else
                L1 = psol.u[j-1][end]
                L2 = psol.u[j][end]
                L3 = psol.u[j+1][end]
                t1 = psol.t[j-1]
                t2 = psol.t[j]
                t3 = psol.t[j+1]
                pde_Lderiv_fd[j-1] = EQL.central_dfdx(L1, L2, L3, t1, t2, t3)
            end
        end

        fig = Figure(fontsize=33)
        times = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        time_indices = [findlast(≤(τ), esol[1].t) for τ in times]
        colors = (:black, :red, :blue, :green, :orange, :purple)
        all_lines = []
        ax = Axis(fig[1, 1],
            xlabel=L"x", ylabel=L"q(x,t)",
            width=600, height=300,
            title=L"(a):$ $ Densities",
            titlealign=:left)
        for (j, i) in pairs(time_indices)
            line = lines!(ax, qstats.knots[i], qstats.means[i], color=colors[j], linewidth=4)
            band!(ax, qstats.knots[i], qstats.lowers[i], qstats.uppers[i], color=(colors[j], 0.3))
            push!(all_lines, line)
            lines!(ax, pde_L[i] * mb_prob.geometry.mesh_points, psol.u[i][begin:(end-1)], linestyle=:dash, color=colors[j], linewidth=4)
        end
        axislegend(ax, all_lines, [L"0", L"10", L"20", L"30", L"40", L"50"], L"t", labelsize=33)
        xlims!(ax, 0, 175)
        ax = Axis(fig[1, 2],
            xlabel=L"t", ylabel=L"L(t)",
            width=600, height=300,
            title=L"(b):$ $ Leading edges",
            titlealign=:left)
        lines!(ax, esol[1].t, Lstats.means, color=:black, label=L"L(t)", linewidth=6)
        lines!(ax, psol.t, pde_L, color=:red, label=L"L_p(t)", linewidth=4, linestyle=:dash)
        axislegend(ax, position=:lt)
        ax = Axis(fig[2, 1],
            xlabel=L"t", ylabel=L"L'(t)",
            width=600, height=300,
            title=L"(c): $ $ Derivatives with average $L(t)$",
            titlealign=:left)
        lines!(ax, esol[1].t[2:end], discrete_Lderivs, color=:black, label=L"$L'(t)$: Average", linewidth=4)
        lines!(ax, psol.t[2:end], pde_Lderivs, color=:red, label=L"$L_p'(t)$ (ITP)", linewidth=4)
        lines!(ax, psol.t[2:end], pde_Lderiv_fd, color=:green, linestyle=:dash, label=L"$L_p'(t)$ (FD)", linewidth=4)
        axislegend(ax, position=:rt)
        ax = Axis(fig[2, 2],
            xlabel=L"t", ylabel=L"L'(t)",
            width=600, height=300,
            title=L"(c): $ $ Derivatives with extrema $L(t)$",
            titlealign=:left)
        lines!(ax, esol[1].t[2:end], discrete_Lderivs2, color=:black, label=L"$L'(t)$: Extrema", linewidth=8)
        lines!(ax, psol.t[2:end], pde_Lderivs, color=:red, label=L"$L_p'(t)$ (ITP)", linewidth=4)
        lines!(ax, psol.t[2:end], pde_Lderiv_fd, color=:green, linestyle=:dash, label=L"$L_p'(t)$ (FD)", linewidth=4)
        axislegend(ax, position=:rt)
        resize_to_layout!(fig)
        fig

        @test_reference joinpath(@__DIR__, "leading_edge_derivatives_proliferation.png") fig
    end

    @testset "MovingBoundaryProblem without Proliferation" begin
        final_time = 14.0
        domain_length = 30.0
        midpoint = domain_length / 2
        cell_nodes = [LinRange(0, midpoint, 10); LinRange(midpoint, domain_length, 20)] |> unique!
        damping_constant = 1.0
        resting_spring_length = 5.0
        spring_constant = 50.0
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
        sol = solve(prob, Tsit5(), saveat=0.01)
        mb_prob = MBProblem(prob, 2500, proliferation=false)
        psol = solve(mb_prob, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t, reltol=1e-9, abstol=1e-9)
        pde_L = psol[end, :]
        discrete_Lderivs = [EQL.cell_dLdt(sol, j) for j in 2:length(sol)]
        pde_Lderivs = [psol(psol.t[j], Val{1}; idxs=length(psol.u[j])) for j in 2:length(psol.t)]
        pde_Lderiv_fd = zeros(length(psol) - 1)
        for j in 2:length(psol.t)
            if j == length(psol.t)
                L1 = psol.u[j-2][end]
                L2 = psol.u[j-1][end]
                L3 = psol.u[j][end]
                t1 = psol.t[j-2]
                t2 = psol.t[j-1]
                t3 = psol.t[j]
                pde_Lderiv_fd[j-1] = EQL.backward_dfdx(L1, L2, L3, t1, t2, t3)
            else
                L1 = psol.u[j-1][end]
                L2 = psol.u[j][end]
                L3 = psol.u[j+1][end]
                t1 = psol.t[j-1]
                t2 = psol.t[j]
                t3 = psol.t[j+1]
                pde_Lderiv_fd[j-1] = EQL.central_dfdx(L1, L2, L3, t1, t2, t3)
            end
        end

        fig = Figure(fontsize=33)
        times = [0.0, 2, 4, 6, 8, 10, 12, 14]
        time_indices = [findlast(≤(τ), sol.t) for τ in times]
        colors = (:black, :red, :blue, :green, :orange, :purple, :brown, :pink)
        all_lines = []
        ax = Axis(fig[1, 1],
            xlabel=L"x", ylabel=L"q(x,t)",
            width=600, height=300,
            title=L"(a):$ $ Densities",
            titlealign=:left)
        for (j, i) in pairs(time_indices)
            line = lines!(ax, sol.u[i], node_densities(sol.u[i]), color=colors[j], linewidth=4)
            push!(all_lines, line)
            lines!(ax, pde_L[i] * mb_prob.geometry.mesh_points, psol.u[i][begin:(end-1)], linestyle=:dash, color=colors[j], linewidth=4)
        end
        axislegend(ax, all_lines, [L"0", L"2", L"4", L"6", L"8", L"10", L"12", L"14"], L"t", labelsize=26)
        xlims!(ax, 0, 175)
        ax = Axis(fig[1, 2],
            xlabel=L"t", ylabel=L"L(t)",
            width=600, height=300,
            title=L"(b):$ $ Leading edges",
            titlealign=:left)
        lines!(ax, sol.t, last.(sol.u), color=:black, label=L"L(t)", linewidth=6)
        lines!(ax, psol.t, pde_L, color=:red, label=L"L_p(t)", linewidth=4, linestyle=:dash)
        axislegend(ax, position=:lt)
        ax = Axis(fig[1, 3],
            xlabel=L"t", ylabel=L"L'(t)",
            width=600, height=300,
            title=L"(c): $ $ Derivatives",
            titlealign=:left)
        lines!(ax, sol.t[2:end], discrete_Lderivs, color=:black, label=L"$L'(t)$", linewidth=8)
        lines!(ax, psol.t[2:end], pde_Lderivs, color=:red, label=L"$L_p'(t)$ (ITP)", linewidth=4)
        lines!(ax, psol.t[2:end], pde_Lderiv_fd, color=:green, linestyle=:dash, label=L"$L_p'(t)$ (FD)", linewidth=4)
        axislegend(ax, position=:rt)
        resize_to_layout!(fig)
        fig

        @test_reference joinpath(@__DIR__, "leading_edge_derivatives_no_proliferation.png") fig
    end
end