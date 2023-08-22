using ..StepwiseEQL
using EpithelialDynamics1D
using OrdinaryDiffEq
using LinearAlgebra
using Bijections
using ElasticArrays
using DataInterpolations
using FiniteVolumeMethod1D
using LinearSolve
using Setfield
using StatsBase
using StableRNGs
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
    model = EQL.EQLModel(sol;
        diffusion_basis, diffusion_parameters)
    model.indicators[2] = false

    add1_results = EQL.add1(model; rng=StableRNG(123))
    @test length(add1_results) == 1 && add1_results[2] == EQL.evaluate_model_loss(@set(model.indicators = [true, true, true]), rng=StableRNG(123))
    drop1_results = EQL.drop1(model; rng=StableRNG(123))
    rng = StableRNG(123)
    subsets = EQL.get_training_and_test_subsets(model; rng)
    @test length(drop1_results) == 2 &&
          drop1_results[1] == EQL.evaluate_model_loss(@set(model.indicators = [false, false, true]), subsets=subsets) &&
          drop1_results[3] == EQL.evaluate_model_loss(@set(model.indicators = [true, false, false]), subsets=subsets)

    model = EQL.EQLModel(sol;
        diffusion_basis, diffusion_parameters)
    model.indicators[1] = false
    results = EQL.drop1(model; rng=StableRNG(123))
    merge!(results, EQL.add1(model; rng=StableRNG(123)))
    subsets = EQL.get_training_and_test_subsets(model; rng=StableRNG(123))
    @test length(results) == 3 &&
          results[3] == EQL.evaluate_model_loss(@set(model.indicators = [false, true, false]), subsets=subsets) &&
          results[2] == EQL.evaluate_model_loss(@set(model.indicators = [false, false, true]), subsets=subsets) &&
          results[1] == EQL.evaluate_model_loss(@set(model.indicators = [true, true, true]), subsets=subsets)
    @inferred EQL.drop1(model, model.indicators, default_loss(; regression=false); rng=StableRNG(123))

    model = EQL.EQLModel(sol;
        diffusion_basis, diffusion_parameters)
    model.indicators[1] = false
    results = EQL.drop1(model; rng=StableRNG(123), skip=Set{Int}(2))
    merge!(results, EQL.add1(model; rng=StableRNG(123), skip=Set{Int}(2)))
    subsets = EQL.get_training_and_test_subsets(model; rng=StableRNG(123))
    @test length(results) == 2 &&
          results[3] == EQL.evaluate_model_loss(@set(model.indicators = [false, true, false]), subsets=subsets) &&
          results[1] == EQL.evaluate_model_loss(@set(model.indicators = [true, true, true]), subsets=subsets)
    @inferred EQL.add1(model; rng=StableRNG(123))

    model = EQL.EQLModel(sol;
        diffusion_basis, diffusion_parameters)
    model.indicators[1] = false
    results = EQL.drop1(model, model.indicators, default_loss(; regression=false); rng=StableRNG(123), skip=Set{Int}(2))
    merge!(results, EQL.add1(model, model.indicators, default_loss(; regression=false); rng=StableRNG(123), skip=Set{Int}(2)))
    subsets = EQL.get_training_and_test_subsets(model; rng=StableRNG(123))
    @test length(results) == 2 &&
          results[3] == EQL.evaluate_model_loss(model, [false, true, false], default_loss(; regression=false), subsets=subsets) &&
          results[1] == EQL.evaluate_model_loss(model, [true, true, true], default_loss(; regression=false), subsets=subsets)

    model = EQL.EQLModel(sol;
        diffusion_basis, diffusion_parameters)
    model.indicators[1] = false
    results = EQL.drop1(model, model.indicators, default_loss(; density=false); rng=StableRNG(123), skip=Set{Int}(2))
    merge!(results, EQL.add1(model, model.indicators, default_loss(; density=false); rng=StableRNG(123), skip=Set{Int}(2)))
    subsets = EQL.get_training_and_test_subsets(model; rng=StableRNG(123))
    @test length(results) == 2 &&
          results[3] == EQL.evaluate_model_loss(model, [false, true, false], default_loss(; density=false), subsets=subsets) &&
          results[1] == EQL.evaluate_model_loss(model, [true, true, true], default_loss(; density=false), subsets=subsets)
end

@testset "Reaction" begin
    final_time = 2.5
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
        (u, β) -> β * u,
        (u, β) -> β * u^2,
        (u, β) -> β * u^3
    )
    reaction_parameters = Gp.β
    emodel = EQL.EQLModel(esol, mesh_points=500;
        diffusion_basis, diffusion_parameters,
        reaction_parameters, reaction_basis,
        threshold_tol=(q=1e-3, dt=1e-1))
    model = EQL.EQLModel(esol[15], mesh_points=1000;
        diffusion_basis, diffusion_parameters,
        reaction_parameters, reaction_basis,
        threshold_tol=(q=1e-3, dt=1e-1))
    model.indicators[[2, 5]] .= false
    emodel.indicators[[3, 4, 5]] .= false
    for regression in (false, true)
        for density in (false, true)
            subsets = EQL.get_training_and_test_subsets(model, rng=StableRNG(299991))
            results = EQL.drop1(model, model.indicators, default_loss(; density, regression); subsets)
            merge!(results, EQL.add1(model, model.indicators, default_loss(; density, regression); subsets))
            @test length(results) == 6 && # [true, false, true, true, false, true]
                  results[5] == EQL.evaluate_model_loss(model, [true, false, true, true, true, true], default_loss(; density=density, regression=regression), subsets=subsets) &&
                  results[1] == EQL.evaluate_model_loss(model, [false, false, true, true, false, true], default_loss(; density=density, regression=regression), subsets=subsets) &&
                  results[2] == EQL.evaluate_model_loss(model, [true, true, true, true, false, true], default_loss(; density=density, regression=regression), subsets=subsets) &&
                  results[3] == EQL.evaluate_model_loss(model, [true, false, false, true, false, true], default_loss(; density=density, regression=regression), subsets=subsets) &&
                  results[6] == EQL.evaluate_model_loss(model, [true, false, true, true, false, false], default_loss(; density=density, regression=regression), subsets=subsets) &&
                  results[4] == EQL.evaluate_model_loss(model, [true, false, true, false, false, true], default_loss(; density=density, regression=regression), subsets=subsets)

            subsets = EQL.get_training_and_test_subsets(emodel, rng=StableRNG(2912325229991))
            results = EQL.drop1(emodel, emodel.indicators, default_loss(; density, regression); subsets)
            merge!(results, EQL.add1(emodel, emodel.indicators, default_loss(; density, regression); subsets))
            @test length(results) == 6 &&
                  results[5] == EQL.evaluate_model_loss(emodel, [true, true, false, false, true, true], default_loss(; density=density, regression=regression), subsets=subsets) &&
                  results[4] == EQL.evaluate_model_loss(emodel, [true, true, false, true, false, true], default_loss(; density=density, regression=regression), subsets=subsets) &&
                  results[1] == EQL.evaluate_model_loss(emodel, [false, true, false, false, false, true], default_loss(; density=density, regression=regression), subsets=subsets) &&
                  results[2] == EQL.evaluate_model_loss(emodel, [true, false, false, false, false, true], default_loss(; density=density, regression=regression), subsets=subsets) &&
                  results[6] == EQL.evaluate_model_loss(emodel, [true, true, false, false, false, false], default_loss(; density=density, regression=regression), subsets=subsets) &&
                  results[3] == EQL.evaluate_model_loss(emodel, [true, true, true, false, false, true], default_loss(; density=density, regression=regression), subsets=subsets)

            subsets = EQL.get_training_and_test_subsets(model, rng=StableRNG(29646346439991))
            results = EQL.drop1(model, model.indicators, default_loss(; density, regression); subsets, skip=Set{Int}((1, 2, 3)))
            merge!(results, EQL.add1(model, model.indicators, default_loss(; density, regression); subsets, skip=Set{Int}((1, 2, 3))))
            @test length(results) == 3 &&
                  results[4] == EQL.evaluate_model_loss(model, [true, false, true, false, false, true], default_loss(; density=density, regression=regression), subsets=subsets) &&
                  results[5] == EQL.evaluate_model_loss(model, [true, false, true, true, true, true], default_loss(; density=density, regression=regression), subsets=subsets) &&
                  results[6] == EQL.evaluate_model_loss(model, [true, false, true, true, false, false], default_loss(; density=density, regression=regression), subsets=subsets)

            subsets = EQL.get_training_and_test_subsets(emodel, rng=StableRNG(2992343991))
            results = EQL.add1(emodel, emodel.indicators, default_loss(; density, regression); subsets, skip=Set{Int}((4, 6, 1)))
            merge!(results, EQL.drop1(emodel, emodel.indicators, default_loss(; density, regression); subsets, skip=Set{Int}((4, 6, 1))))
            @test length(results) == 3 &&
                  results[5] == EQL.evaluate_model_loss(emodel, [true, true, false, false, true, true], default_loss(; density, regression); subsets=subsets) &&
                  results[2] == EQL.evaluate_model_loss(emodel, [true, false, false, false, false, true], default_loss(; density, regression); subsets=subsets) &&
                  results[3] == EQL.evaluate_model_loss(emodel, [true, true, true, false, false, true], default_loss(; density, regression); subsets=subsets)
        end
    end
end