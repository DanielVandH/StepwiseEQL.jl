using ..StepwiseEQL
using EpithelialDynamics1D
using OrdinaryDiffEq
using LinearAlgebra
using Bijections
using ElasticArrays
using MovingBoundaryProblems1D
using FiniteVolumeMethod1D
using LinearSolve
using Setfield
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

    @test !EQL.has_proliferation(sol)
    @test !EQL.has_moving_boundary(sol)
    @test EQL.get_saveat(sol) == sol.t

    # Getting a template
    pde = EQL.build_pde(sol, 1000;
        diffusion_basis, diffusion_parameters, diffusion_theta=[0.0, 0.0, 0.0])
    pde_sol = solve(pde, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    _pde = FVMProblem(prob, 1000;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.0, 0.0, 0.0], p=nothing),
        proliferation=false)
    _pde_sol = solve(_pde, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    @test pde_sol.u == _pde_sol.u
    @test pde_sol.u == solve(model.pde_template, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t).u

    # Rebuilding: Zeros
    pde = EQL.rebuild_pde(model)
    pde_sol = solve(pde, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    _pde = FVMProblem(prob, 1000;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.0, 0.0, 0.0], p=nothing),
        proliferation=false)
    _pde_sol = solve(_pde, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    @test pde_sol.u == _pde_sol.u

    # Rebuilding: A new vector 
    pde = EQL.rebuild_pde(model, [0.7, 0.5, 0.1])
    pde_sol = solve(pde, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    _pde = FVMProblem(prob, 1000;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.7, 0.5, 0.1], p=nothing),
        proliferation=false)
    _pde_sol = solve(_pde, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    @test pde_sol.u == _pde_sol.u

    # Rebuilding: Partial model 
    model.indicators[2] = false
    pde = EQL.rebuild_pde(model, [0.7, 0.5, 0.1]) # keeping [0.5] to make sure it's ignored
    pde_sol = solve(pde, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    _pde = FVMProblem(prob, 1000;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.7, 0.0, 0.1], p=nothing),
        proliferation=false)
    _pde_sol = solve(_pde, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    @test pde_sol.u == _pde_sol.u
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
    esol = solve(ens_prob, Tsit5(); trajectories=10, saveat=5.0)
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
    emodel = EQL.EQLModel(esol, mesh_points=250;
        diffusion_basis, diffusion_parameters,
        reaction_parameters, reaction_basis)
    emodel_fixed_diffusion = EQL.EQLModel(esol, mesh_points=500;
        diffusion_basis, diffusion_parameters,
        reaction_parameters, reaction_basis,
        diffusion_theta=[0.1, 0.2, 0.3])
    emodel_fixed_reaction = EQL.EQLModel(esol;
        diffusion_basis, diffusion_parameters,
        reaction_parameters, reaction_basis,
        reaction_theta=[0.17, 0.27, 0.32])
    model = EQL.EQLModel(sol, mesh_points=250;
        diffusion_basis, diffusion_parameters,
        reaction_parameters, reaction_basis)
    model_fixed_diffusion = EQL.EQLModel(sol, mesh_points=500;
        diffusion_basis, diffusion_parameters,
        reaction_parameters, reaction_basis,
        diffusion_theta=[0.1, 0.2, 0.3])
    model_fixed_reaction = EQL.EQLModel(sol;
        diffusion_basis, diffusion_parameters,
        reaction_parameters, reaction_basis,
        reaction_theta=[0.17, 0.27, 0.32])

    @test EQL.has_proliferation(esol)
    @test EQL.has_proliferation(sol)
    @test !EQL.has_moving_boundary(esol)
    @test !EQL.has_moving_boundary(sol)
    @test EQL.get_saveat(esol) == esol[1].t
    @test EQL.get_saveat(sol) == sol.t
    @test emodel.indicators == [true, true, true, true, true, true]
    @test emodel_fixed_diffusion.indicators == [true, true, true]
    @test emodel_fixed_reaction.indicators == [true, true, true]
    @test model.indicators == [true, true, true, true, true, true]
    @test model_fixed_diffusion.indicators == [true, true, true]
    @test model_fixed_reaction.indicators == [true, true, true]

    # Getting a template
    pde_from_esol = EQL.build_pde(esol, 250;
        diffusion_basis, diffusion_parameters,
        reaction_basis, reaction_parameters)
    pde_fixed_diffusion_from_esol = EQL.build_pde(esol, 500;
        diffusion_basis, diffusion_parameters,
        reaction_basis, reaction_parameters,
        diffusion_theta=[0.1, 0.2, 0.3])
    pde_fixed_reaction_from_esol = EQL.build_pde(esol, 1000;
        diffusion_basis, diffusion_parameters,
        reaction_basis, reaction_parameters,
        reaction_theta=[0.17, 0.27, 0.32])
    pde_from_sol = EQL.build_pde(sol, 250;
        diffusion_basis, diffusion_parameters,
        reaction_basis, reaction_parameters)
    pde_fixed_diffusion_from_sol = EQL.build_pde(sol, 500;
        diffusion_basis, diffusion_parameters,
        reaction_basis, reaction_parameters,
        diffusion_theta=[0.1, 0.2, 0.3])
    pde_fixed_reaction_from_sol = EQL.build_pde(sol, 1000;
        diffusion_basis, diffusion_parameters,
        reaction_basis, reaction_parameters,
        reaction_theta=[0.17, 0.27, 0.32])
    manual_pde_from_esol = FVMProblem(prob, 250;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.0, 0.0, 0.0], p=k),
        reaction_function=reaction_basis,
        reaction_parameters=EQL.Parameters(θ=[0.0, 0.0, 0.0], p=Gp.β),
        proliferation=true)
    manual_pde_from_sol = FVMProblem(prob, 250;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.0, 0.0, 0.0], p=k),
        reaction_function=reaction_basis,
        reaction_parameters=EQL.Parameters(θ=[0.0, 0.0, 0.0], p=Gp.β),
        proliferation=true)
    manual_pde_fixed_diffusion_from_esol = FVMProblem(prob, 500;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.1, 0.2, 0.3], p=k),
        reaction_function=reaction_basis,
        reaction_parameters=EQL.Parameters(θ=[0.0, 0.0, 0.0], p=Gp.β),
        proliferation=true)
    manual_pde_fixed_diffusion_from_sol = FVMProblem(prob, 500;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.1, 0.2, 0.3], p=k),
        reaction_function=reaction_basis,
        reaction_parameters=EQL.Parameters(θ=[0.0, 0.0, 0.0], p=Gp.β),
        proliferation=true)
    manual_pde_fixed_reaction_from_esol = FVMProblem(prob, 1000;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.0, 0.0, 0.0], p=k),
        reaction_function=reaction_basis,
        reaction_parameters=EQL.Parameters(θ=[0.17, 0.27, 0.32], p=Gp.β),
        proliferation=true)
    manual_pde_fixed_reaction_from_sol = FVMProblem(prob, 1000;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.0, 0.0, 0.0], p=k),
        reaction_function=reaction_basis,
        reaction_parameters=EQL.Parameters(θ=[0.17, 0.27, 0.32], p=Gp.β),
        proliferation=true)
    pde_from_esol_sol = solve(pde_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_fixed_diffusion_from_esol_sol = solve(pde_fixed_diffusion_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_fixed_reaction_from_esol_sol = solve(pde_fixed_reaction_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_from_sol_sol = solve(pde_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_fixed_diffusion_from_sol_sol = solve(pde_fixed_diffusion_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_fixed_reaction_from_sol_sol = solve(pde_fixed_reaction_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    manual_pde_from_esol_sol = solve(manual_pde_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    manual_pde_fixed_diffusion_from_esol_sol = solve(manual_pde_fixed_diffusion_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    manual_pde_fixed_reaction_from_esol_sol = solve(manual_pde_fixed_reaction_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    manual_pde_from_sol_sol = solve(manual_pde_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    manual_pde_fixed_diffusion_from_sol_sol = solve(manual_pde_fixed_diffusion_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    manual_pde_fixed_reaction_from_sol_sol = solve(manual_pde_fixed_reaction_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    @test pde_from_esol_sol.u == manual_pde_from_esol_sol.u
    @test pde_fixed_diffusion_from_esol_sol.u == manual_pde_fixed_diffusion_from_esol_sol.u
    @test pde_fixed_reaction_from_esol_sol.u == manual_pde_fixed_reaction_from_esol_sol.u
    @test pde_from_sol_sol.u == manual_pde_from_sol_sol.u
    @test pde_fixed_diffusion_from_sol_sol.u == manual_pde_fixed_diffusion_from_sol_sol.u
    @test pde_fixed_reaction_from_sol_sol.u == manual_pde_fixed_reaction_from_sol_sol.u

    # Rebuilding: Zeros
    pde_from_esol = EQL.rebuild_pde(emodel)
    pde_fixed_diffusion_from_esol = EQL.rebuild_pde(emodel_fixed_diffusion)
    pde_fixed_reaction_from_esol = EQL.rebuild_pde(emodel_fixed_reaction)
    pde_from_sol = EQL.rebuild_pde(model)
    pde_fixed_diffusion_from_sol = EQL.rebuild_pde(model_fixed_diffusion)
    pde_fixed_reaction_from_sol = EQL.rebuild_pde(model_fixed_reaction)
    pde_from_esol_sol = solve(pde_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_fixed_diffusion_from_esol_sol = solve(pde_fixed_diffusion_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_fixed_reaction_from_esol_sol = solve(pde_fixed_reaction_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_from_sol_sol = solve(pde_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_fixed_diffusion_from_sol_sol = solve(pde_fixed_diffusion_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_fixed_reaction_from_sol_sol = solve(pde_fixed_reaction_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    @test pde_from_esol_sol.u == manual_pde_from_esol_sol.u
    @test pde_fixed_diffusion_from_esol_sol.u == manual_pde_fixed_diffusion_from_esol_sol.u
    @test pde_fixed_reaction_from_esol_sol.u == manual_pde_fixed_reaction_from_esol_sol.u
    @test pde_from_sol_sol.u == manual_pde_from_sol_sol.u
    @test pde_fixed_diffusion_from_sol_sol.u == manual_pde_fixed_diffusion_from_sol_sol.u
    @test pde_fixed_reaction_from_sol_sol.u == manual_pde_fixed_reaction_from_sol_sol.u

    # Rebuilding: A new vector
    pde_from_esol = EQL.rebuild_pde(emodel, [0.3, 0.5, 2.5, 6.6, 0.2, 0.5])
    pde_fixed_diffusion_from_esol = EQL.rebuild_pde(emodel_fixed_diffusion, [6.6, 0.2, 0.5])
    pde_fixed_reaction_from_esol = EQL.rebuild_pde(emodel_fixed_reaction, [0.3, 0.5, 2.5])
    pde_from_sol = EQL.rebuild_pde(model, [0.3, 0.5, 2.5, 6.6, 0.2, 0.5])
    pde_fixed_diffusion_from_sol = EQL.rebuild_pde(model_fixed_diffusion, [6.6, 0.2, 0.5])
    pde_fixed_reaction_from_sol = EQL.rebuild_pde(model_fixed_reaction, [0.3, 0.5, 2.5])
    pde_from_esol_sol = solve(pde_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_fixed_diffusion_from_esol_sol = solve(pde_fixed_diffusion_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_fixed_reaction_from_esol_sol = solve(pde_fixed_reaction_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_from_sol_sol = solve(pde_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_fixed_diffusion_from_sol_sol = solve(pde_fixed_diffusion_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_fixed_reaction_from_sol_sol = solve(pde_fixed_reaction_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    manual_pde_from_esol = FVMProblem(prob, 250;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.3, 0.5, 2.5], p=k),
        reaction_function=reaction_basis,
        reaction_parameters=EQL.Parameters(θ=[6.6, 0.2, 0.5], p=Gp.β),
        proliferation=true)
    manual_pde_from_sol = FVMProblem(prob, 250;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.3, 0.5, 2.5], p=k),
        reaction_function=reaction_basis,
        reaction_parameters=EQL.Parameters(θ=[6.6, 0.2, 0.5], p=Gp.β),
        proliferation=true)
    manual_pde_fixed_diffusion_from_esol = FVMProblem(prob, 500;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.1, 0.2, 0.3], p=k),
        reaction_function=reaction_basis,
        reaction_parameters=EQL.Parameters(θ=[6.6, 0.2, 0.5], p=Gp.β),
        proliferation=true)
    manual_pde_fixed_diffusion_from_sol = FVMProblem(prob, 500;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.1, 0.2, 0.3], p=k),
        reaction_function=reaction_basis,
        reaction_parameters=EQL.Parameters(θ=[6.6, 0.2, 0.5], p=Gp.β),
        proliferation=true)
    manual_pde_fixed_reaction_from_esol = FVMProblem(prob, 1000;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.3, 0.5, 2.5], p=k),
        reaction_function=reaction_basis,
        reaction_parameters=EQL.Parameters(θ=[0.17, 0.27, 0.32], p=Gp.β),
        proliferation=true)
    manual_pde_fixed_reaction_from_sol = FVMProblem(prob, 1000;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.3, 0.5, 2.5], p=k),
        reaction_function=reaction_basis,
        reaction_parameters=EQL.Parameters(θ=[0.17, 0.27, 0.32], p=Gp.β),
        proliferation=true)
    manual_pde_from_esol_sol = solve(manual_pde_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    manual_pde_fixed_diffusion_from_esol_sol = solve(manual_pde_fixed_diffusion_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    manual_pde_fixed_reaction_from_esol_sol = solve(manual_pde_fixed_reaction_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    manual_pde_from_sol_sol = solve(manual_pde_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    manual_pde_fixed_diffusion_from_sol_sol = solve(manual_pde_fixed_diffusion_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    manual_pde_fixed_reaction_from_sol_sol = solve(manual_pde_fixed_reaction_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    @test pde_from_esol_sol.u == manual_pde_from_esol_sol.u
    @test pde_fixed_diffusion_from_esol_sol.u == manual_pde_fixed_diffusion_from_esol_sol.u
    @test pde_fixed_reaction_from_esol_sol.u == manual_pde_fixed_reaction_from_esol_sol.u
    @test pde_from_sol_sol.u == manual_pde_from_sol_sol.u
    @test pde_fixed_diffusion_from_sol_sol.u == manual_pde_fixed_diffusion_from_sol_sol.u
    @test pde_fixed_reaction_from_sol_sol.u == manual_pde_fixed_reaction_from_sol_sol.u

    # Rebuilding: Partial model 
    emodel.indicators[[2, 5]] .= false
    emodel_fixed_diffusion.indicators[[2]] .= false
    emodel_fixed_reaction.indicators[[3]] .= false
    model.indicators[[1, 2, 6]] .= false
    model_fixed_diffusion.indicators[[1, 3]] .= false
    model_fixed_reaction.indicators[[2, 3]] .= false
    pde_from_esol = EQL.rebuild_pde(emodel, [0.3, 0.0, 2.5, 6.6, 0.0, 0.5])
    pde_fixed_diffusion_from_esol = EQL.rebuild_pde(emodel_fixed_diffusion, [6.6, 0.0, 0.5])
    pde_fixed_reaction_from_esol = EQL.rebuild_pde(emodel_fixed_reaction, [0.3, 0.5, 0.0])
    pde_from_sol = EQL.rebuild_pde(model, [0.0, 0.0, 2.5, 6.6, 0.2, 0.0])
    pde_fixed_diffusion_from_sol = EQL.rebuild_pde(model_fixed_diffusion, [0.0, 0.2, 0.0])
    pde_fixed_reaction_from_sol = EQL.rebuild_pde(model_fixed_reaction, [0.3, 0.0, 0.0])
    manual_pde_from_esol = FVMProblem(prob, 250;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.3, 0.0, 2.5], p=k),
        reaction_function=reaction_basis,
        reaction_parameters=EQL.Parameters(θ=[6.6, 0.0, 0.5], p=Gp.β),
        proliferation=true)
    manual_pde_from_sol = FVMProblem(prob, 250;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.0, 0.0, 2.5], p=k),
        reaction_function=reaction_basis,
        reaction_parameters=EQL.Parameters(θ=[6.6, 0.2, 0.0], p=Gp.β),
        proliferation=true)
    manual_pde_fixed_diffusion_from_esol = FVMProblem(prob, 500;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.1, 0.2, 0.3], p=k),
        reaction_function=reaction_basis,
        reaction_parameters=EQL.Parameters(θ=[6.6, 0.0, 0.5], p=Gp.β),
        proliferation=true)
    manual_pde_fixed_diffusion_from_sol = FVMProblem(prob, 500;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.1, 0.2, 0.3], p=k),
        reaction_function=reaction_basis,
        reaction_parameters=EQL.Parameters(θ=[0.0, 0.2, 0.0], p=Gp.β),
        proliferation=true)
    manual_pde_fixed_reaction_from_esol = FVMProblem(prob, 1000;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.3, 0.5, 0.0], p=k),
        reaction_function=reaction_basis,
        reaction_parameters=EQL.Parameters(θ=[0.17, 0.27, 0.32], p=Gp.β),
        proliferation=true)
    manual_pde_fixed_reaction_from_sol = FVMProblem(prob, 1000;
        diffusion_function=diffusion_basis,
        diffusion_parameters=EQL.Parameters(θ=[0.3, 0.0, 0.0], p=k),
        reaction_function=reaction_basis,
        reaction_parameters=EQL.Parameters(θ=[0.17, 0.27, 0.32], p=Gp.β),
        proliferation=true)
    pde_from_esol_sol = solve(pde_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_fixed_diffusion_from_esol_sol = solve(pde_fixed_diffusion_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_fixed_reaction_from_esol_sol = solve(pde_fixed_reaction_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_from_sol_sol = solve(pde_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_fixed_diffusion_from_sol_sol = solve(pde_fixed_diffusion_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    pde_fixed_reaction_from_sol_sol = solve(pde_fixed_reaction_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    manual_pde_from_esol_sol = solve(manual_pde_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    manual_pde_fixed_diffusion_from_esol_sol = solve(manual_pde_fixed_diffusion_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    manual_pde_fixed_reaction_from_esol_sol = solve(manual_pde_fixed_reaction_from_esol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    manual_pde_from_sol_sol = solve(manual_pde_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    manual_pde_fixed_diffusion_from_sol_sol = solve(manual_pde_fixed_diffusion_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    manual_pde_fixed_reaction_from_sol_sol = solve(manual_pde_fixed_reaction_from_sol, TRBDF2(linsolve=KLUFactorization()), saveat=sol.t)
    @test pde_from_esol_sol.u == manual_pde_from_esol_sol.u
    @test pde_fixed_diffusion_from_esol_sol.u == manual_pde_fixed_diffusion_from_esol_sol.u
    @test pde_fixed_reaction_from_esol_sol.u == manual_pde_fixed_reaction_from_esol_sol.u
    @test pde_from_sol_sol.u == manual_pde_from_sol_sol.u
    @test pde_fixed_diffusion_from_sol_sol.u == manual_pde_fixed_diffusion_from_sol_sol.u
    @test pde_fixed_reaction_from_sol_sol.u == manual_pde_fixed_reaction_from_sol_sol.u
end

@testset "Moving Boundary without Proliferation" begin
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
        (u, k) -> k * inv(u),
        (u, k) -> k * inv(u^2),
        (u, k) -> k * inv(u^3)
    )
    diffusion_parameters = spring_constant
    rhs_parameters = resting_spring_length
    moving_boundary_parameters = spring_constant
    diffusion_theta = [1e-6, 1.0, 0.0]
    rhs_theta = [1e-6, 2.0, -2.0, 0.0]
    moving_boundary_theta = [0.0, 1.0, 1e-6]

    model = EQL.EQLModel(sol; mesh_points=500, diffusion_basis, diffusion_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters)
    model_fixed_diffusion = EQL.EQLModel(sol; mesh_points=500, diffusion_basis, diffusion_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta)
    model_fixed_rhs = EQL.EQLModel(sol; mesh_points=500, diffusion_basis, diffusion_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, rhs_theta)
    model_fixed_moving_boundary = EQL.EQLModel(sol; mesh_points=500, diffusion_basis, diffusion_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, moving_boundary_theta)
    model_fixed_diffusion_rhs = EQL.EQLModel(sol; mesh_points=500, diffusion_basis, diffusion_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, rhs_theta)
    model_fixed_diffusion_moving_boundary = EQL.EQLModel(sol; mesh_points=500, diffusion_basis, diffusion_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, moving_boundary_theta)
    model_fixed_rhs_moving_boundary = EQL.EQLModel(sol; mesh_points=500, diffusion_basis, diffusion_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, rhs_theta, moving_boundary_theta)

    @test !EQL.has_proliferation(sol)
    @test EQL.get_saveat(sol) == sol.t
    @test model.indicators == [fill(true, length(diffusion_theta)); fill(true, length(rhs_theta)); fill(true, length(moving_boundary_theta))]
    @test model_fixed_diffusion.indicators == [fill(true, length(rhs_theta)); fill(true, length(moving_boundary_theta))]
    @test model_fixed_rhs.indicators == [fill(true, length(diffusion_theta)); fill(true, length(moving_boundary_theta))]
    @test model_fixed_moving_boundary.indicators == [fill(true, length(diffusion_theta)); fill(true, length(rhs_theta))]
    @test model_fixed_diffusion_rhs.indicators == [fill(true, length(moving_boundary_theta));]
    @test model_fixed_diffusion_moving_boundary.indicators == [fill(true, length(rhs_theta));]
    @test model_fixed_rhs_moving_boundary.indicators == [fill(true, length(diffusion_theta));]

    pde_model = EQL.build_pde(sol, 500; diffusion_basis, diffusion_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters)
    pde_model_fixed_diffusion = EQL.build_pde(sol, 500; diffusion_basis, diffusion_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta)
    pde_model_fixed_rhs = EQL.build_pde(sol, 500; diffusion_basis, diffusion_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, rhs_theta)
    pde_model_fixed_moving_boundary = EQL.build_pde(sol, 500; diffusion_basis, diffusion_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, moving_boundary_theta)
    pde_model_fixed_diffusion_rhs = EQL.build_pde(sol, 500; diffusion_basis, diffusion_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, rhs_theta)
    pde_model_fixed_diffusion_moving_boundary = EQL.build_pde(sol, 500; diffusion_basis, diffusion_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, moving_boundary_theta)
    pde_model_fixed_rhs_moving_boundary = EQL.build_pde(sol, 500; diffusion_basis, diffusion_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, rhs_theta, moving_boundary_theta)
    manual_pde_model = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zero(diffusion_theta), p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zero(rhs_theta), p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zero(moving_boundary_theta), p=moving_boundary_parameters), proliferation=false)
    manual_pde_model_fixed_diffusion = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zero(rhs_theta), p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zero(moving_boundary_theta), p=moving_boundary_parameters), proliferation=false)
    manual_pde_model_fixed_rhs = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zero(diffusion_theta), p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zero(moving_boundary_theta), p=moving_boundary_parameters), proliferation=false)
    manual_pde_model_fixed_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zero(diffusion_theta), p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zero(rhs_theta), p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=false)
    manual_pde_model_fixed_diffusion_rhs = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zero(moving_boundary_theta), p=moving_boundary_parameters), proliferation=false)
    manual_pde_model_fixed_diffusion_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zero(rhs_theta), p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=false)
    manual_pde_model_fixed_rhs_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zero(diffusion_theta), p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=false)

    sol_pde_model = solve(pde_model, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_diffusion = solve(pde_model_fixed_diffusion, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_rhs = solve(pde_model_fixed_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_moving_boundary = solve(pde_model_fixed_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_diffusion_rhs = solve(pde_model_fixed_diffusion_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_diffusion_moving_boundary = solve(pde_model_fixed_diffusion_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_rhs_moving_boundary = solve(pde_model_fixed_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model = solve(manual_pde_model, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model_fixed_diffusion = solve(manual_pde_model_fixed_diffusion, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model_fixed_rhs = solve(manual_pde_model_fixed_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model_fixed_moving_boundary = solve(manual_pde_model_fixed_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model_fixed_diffusion_rhs = solve(manual_pde_model_fixed_diffusion_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model_fixed_diffusion_moving_boundary = solve(manual_pde_model_fixed_diffusion_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model_fixed_rhs_moving_boundary = solve(manual_pde_model_fixed_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)

    @test sol_pde_model.u == sol_manual_pde_model.u
    @test sol_pde_model_fixed_diffusion.u == sol_manual_pde_model_fixed_diffusion.u
    @test sol_pde_model_fixed_rhs.u == sol_manual_pde_model_fixed_rhs.u
    @test sol_pde_model_fixed_moving_boundary.u == sol_manual_pde_model_fixed_moving_boundary.u
    @test sol_pde_model_fixed_diffusion_rhs.u == sol_manual_pde_model_fixed_diffusion_rhs.u
    @test sol_pde_model_fixed_diffusion_moving_boundary.u == sol_manual_pde_model_fixed_diffusion_moving_boundary.u
    @test sol_pde_model_fixed_rhs_moving_boundary.u == sol_manual_pde_model_fixed_rhs_moving_boundary.u

    # Rebuilding: Zeros  
    pde_model = EQL.rebuild_pde(model)
    pde_model_fixed_diffusion = EQL.rebuild_pde(model_fixed_diffusion)
    pde_model_fixed_rhs = EQL.rebuild_pde(model_fixed_rhs)
    pde_model_fixed_moving_boundary = EQL.rebuild_pde(model_fixed_moving_boundary)
    pde_model_fixed_diffusion_rhs = EQL.rebuild_pde(model_fixed_diffusion_rhs)
    pde_model_fixed_diffusion_moving_boundary = EQL.rebuild_pde(model_fixed_diffusion_moving_boundary)
    pde_model_fixed_rhs_moving_boundary = EQL.rebuild_pde(model_fixed_rhs_moving_boundary)
    sol_pde_model = solve(pde_model, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_diffusion = solve(pde_model_fixed_diffusion, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_rhs = solve(pde_model_fixed_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_moving_boundary = solve(pde_model_fixed_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_diffusion_rhs = solve(pde_model_fixed_diffusion_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_diffusion_moving_boundary = solve(pde_model_fixed_diffusion_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_rhs_moving_boundary = solve(pde_model_fixed_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    @test sol_pde_model.u ≈ sol_manual_pde_model.u
    @test sol_pde_model_fixed_diffusion.u == sol_manual_pde_model_fixed_diffusion.u
    @test sol_pde_model_fixed_rhs.u == sol_manual_pde_model_fixed_rhs.u
    @test sol_pde_model_fixed_moving_boundary.u == sol_manual_pde_model_fixed_moving_boundary.u
    @test sol_pde_model_fixed_diffusion_rhs.u == sol_manual_pde_model_fixed_diffusion_rhs.u
    @test sol_pde_model_fixed_diffusion_moving_boundary.u == sol_manual_pde_model_fixed_diffusion_moving_boundary.u
    @test sol_pde_model_fixed_rhs_moving_boundary.u == sol_manual_pde_model_fixed_rhs_moving_boundary.u

    # Rebuilding: A new vector
    θd = diffusion_theta + 1e-2randn(length(diffusion_theta))
    θr = rhs_theta + 1e-2randn(length(rhs_theta))
    θm = moving_boundary_theta + 1e-2randn(length(moving_boundary_theta))
    pde_model = EQL.rebuild_pde(model, [θd; θr; θm])
    pde_model_fixed_diffusion = EQL.rebuild_pde(model_fixed_diffusion, [θr; θm])
    pde_model_fixed_rhs = EQL.rebuild_pde(model_fixed_rhs, [θd; θm])
    pde_model_fixed_moving_boundary = EQL.rebuild_pde(model_fixed_moving_boundary, [θd; θr])
    pde_model_fixed_diffusion_rhs = EQL.rebuild_pde(model_fixed_diffusion_rhs, [θm;])
    pde_model_fixed_diffusion_moving_boundary = EQL.rebuild_pde(model_fixed_diffusion_moving_boundary, [θr;])
    pde_model_fixed_rhs_moving_boundary = EQL.rebuild_pde(model_fixed_rhs_moving_boundary, [θd;])
    manual_pde_model = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=θd, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=θr, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=θm, p=moving_boundary_parameters), proliferation=false)
    manual_pde_model_fixed_diffusion = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=θr, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=θm, p=moving_boundary_parameters), proliferation=false)
    manual_pde_model_fixed_rhs = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=θd, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=θm, p=moving_boundary_parameters), proliferation=false)
    manual_pde_model_fixed_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=θd, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=θr, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=false)
    manual_pde_model_fixed_diffusion_rhs = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=θm, p=moving_boundary_parameters), proliferation=false)
    manual_pde_model_fixed_diffusion_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=θr, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=false)
    manual_pde_model_fixed_rhs_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=θd, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=false)
    sol_pde_model = solve(pde_model, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_diffusion = solve(pde_model_fixed_diffusion, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_rhs = solve(pde_model_fixed_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_moving_boundary = solve(pde_model_fixed_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_diffusion_rhs = solve(pde_model_fixed_diffusion_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_diffusion_moving_boundary = solve(pde_model_fixed_diffusion_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_rhs_moving_boundary = solve(pde_model_fixed_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model = solve(manual_pde_model, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model_fixed_diffusion = solve(manual_pde_model_fixed_diffusion, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model_fixed_rhs = solve(manual_pde_model_fixed_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model_fixed_moving_boundary = solve(manual_pde_model_fixed_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model_fixed_diffusion_rhs = solve(manual_pde_model_fixed_diffusion_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model_fixed_diffusion_moving_boundary = solve(manual_pde_model_fixed_diffusion_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model_fixed_rhs_moving_boundary = solve(manual_pde_model_fixed_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    @test sol_pde_model.u == sol_manual_pde_model.u
    @test sol_pde_model_fixed_diffusion.u == sol_manual_pde_model_fixed_diffusion.u
    @test sol_pde_model_fixed_rhs.u == sol_manual_pde_model_fixed_rhs.u
    @test sol_pde_model_fixed_moving_boundary.u == sol_manual_pde_model_fixed_moving_boundary.u
    @test sol_pde_model_fixed_diffusion_rhs.u == sol_manual_pde_model_fixed_diffusion_rhs.u
    @test sol_pde_model_fixed_diffusion_moving_boundary.u == sol_manual_pde_model_fixed_diffusion_moving_boundary.u
    @test sol_pde_model_fixed_rhs_moving_boundary.u == sol_manual_pde_model_fixed_rhs_moving_boundary.u

    # Rebuilding: Partial model 
    θd = diffusion_theta + 1e-2randn(length(diffusion_theta))
    θr = rhs_theta + 1e-2randn(length(rhs_theta))
    θm = moving_boundary_theta + 1e-2randn(length(moving_boundary_theta))
    Id = [false, true, false]
    Ir = [false, true, true, false]
    Im = [false, true, false]
    zdiffusion_theta = copy(θd)
    zdiffusion_theta[.!Id] .= 0.0
    zrhs_theta = copy(θr)
    zrhs_theta[.!Ir] .= 0.0
    zmoving_boundary_theta = copy(θm)
    zmoving_boundary_theta[.!Im] .= 0.0
    pde_model = EQL.rebuild_pde(model, [θd; θr; θm], [Id; Ir; Im])
    pde_model_fixed_diffusion = EQL.rebuild_pde(model_fixed_diffusion, [θr; θm], [Ir; Im])
    pde_model_fixed_rhs = EQL.rebuild_pde(model_fixed_rhs, [θd; θm], [Id; Im])
    pde_model_fixed_moving_boundary = EQL.rebuild_pde(model_fixed_moving_boundary, [θd; θr], [Id; Ir])
    pde_model_fixed_diffusion_rhs = EQL.rebuild_pde(model_fixed_diffusion_rhs, [θm;], [Im;])
    pde_model_fixed_diffusion_moving_boundary = EQL.rebuild_pde(model_fixed_diffusion_moving_boundary, [θr;], [Ir;])
    pde_model_fixed_rhs_moving_boundary = EQL.rebuild_pde(model_fixed_rhs_moving_boundary, [θd;], [Id;])
    manual_pde_model = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zdiffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zrhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zmoving_boundary_theta, p=moving_boundary_parameters), proliferation=false)
    manual_pde_model_fixed_diffusion = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zrhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zmoving_boundary_theta, p=moving_boundary_parameters), proliferation=false)
    manual_pde_model_fixed_rhs = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zdiffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zmoving_boundary_theta, p=moving_boundary_parameters), proliferation=false)
    manual_pde_model_fixed_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zdiffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zrhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=false)
    manual_pde_model_fixed_diffusion_rhs = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zmoving_boundary_theta, p=moving_boundary_parameters), proliferation=false)
    manual_pde_model_fixed_diffusion_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zrhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=false)
    manual_pde_model_fixed_rhs_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zdiffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=false)
    sol_pde_model = solve(pde_model, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_diffusion = solve(pde_model_fixed_diffusion, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_rhs = solve(pde_model_fixed_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_moving_boundary = solve(pde_model_fixed_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_diffusion_rhs = solve(pde_model_fixed_diffusion_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_diffusion_moving_boundary = solve(pde_model_fixed_diffusion_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_pde_model_fixed_rhs_moving_boundary = solve(pde_model_fixed_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model = solve(manual_pde_model, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model_fixed_diffusion = solve(manual_pde_model_fixed_diffusion, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model_fixed_rhs = solve(manual_pde_model_fixed_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model_fixed_moving_boundary = solve(manual_pde_model_fixed_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model_fixed_diffusion_rhs = solve(manual_pde_model_fixed_diffusion_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model_fixed_diffusion_moving_boundary = solve(manual_pde_model_fixed_diffusion_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    sol_manual_pde_model_fixed_rhs_moving_boundary = solve(manual_pde_model_fixed_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=0.1)
    @test sol_pde_model.u == sol_manual_pde_model.u
    @test sol_pde_model_fixed_diffusion.u == sol_manual_pde_model_fixed_diffusion.u
    @test sol_pde_model_fixed_rhs.u == sol_manual_pde_model_fixed_rhs.u
    @test sol_pde_model_fixed_moving_boundary.u == sol_manual_pde_model_fixed_moving_boundary.u
    @test sol_pde_model_fixed_diffusion_rhs.u == sol_manual_pde_model_fixed_diffusion_rhs.u
    @test sol_pde_model_fixed_diffusion_moving_boundary.u == sol_manual_pde_model_fixed_diffusion_moving_boundary.u
    @test sol_pde_model_fixed_rhs_moving_boundary.u == sol_manual_pde_model_fixed_rhs_moving_boundary.u
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
        (u, k) -> k * inv(u),
        (u, k) -> k * inv(u^2),
        (u, k) -> k * inv(u^3)
    )
    diffusion_parameters = k
    reaction_parameters = β
    rhs_parameters = resting_spring_length
    moving_boundary_parameters = spring_constant
    diffusion_theta = [1e-6, 1.0, 0.0]
    reaction_theta = [0.0, 1.0, -1.0]
    rhs_theta = [1e-6, 2.0, -2.0, 0.0]
    moving_boundary_theta = [0.0, 1.0, 1e-6]

    ensemble_model = EQL.EQLModel(ensemble_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters)
    ensemble_model_fixed_diffusion = EQL.EQLModel(ensemble_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta)
    ensemble_model_fixed_reaction = EQL.EQLModel(ensemble_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta)
    ensemble_model_fixed_rhs = EQL.EQLModel(ensemble_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, rhs_theta)
    ensemble_model_fixed_moving_boundary = EQL.EQLModel(ensemble_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, moving_boundary_theta)
    ensemble_model_fixed_diffusion_reaction = EQL.EQLModel(ensemble_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta)
    ensemble_model_fixed_diffusion_rhs = EQL.EQLModel(ensemble_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, rhs_theta)
    ensemble_model_fixed_diffusion_moving_boundary = EQL.EQLModel(ensemble_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, moving_boundary_theta)
    ensemble_model_fixed_reaction_rhs = EQL.EQLModel(ensemble_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, rhs_theta)
    ensemble_model_fixed_reaction_moving_boundary = EQL.EQLModel(ensemble_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, moving_boundary_theta)
    ensemble_model_fixed_rhs_moving_boundary = EQL.EQLModel(ensemble_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, rhs_theta, moving_boundary_theta)
    ensemble_model_fixed_diffusion_reaction_rhs = EQL.EQLModel(ensemble_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta, rhs_theta)
    ensemble_model_fixed_diffusion_reaction_moving_boundary = EQL.EQLModel(ensemble_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta, moving_boundary_theta)
    ensemble_model_fixed_diffusion_rhs_moving_boundary = EQL.EQLModel(ensemble_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, rhs_theta, moving_boundary_theta)
    ensemble_model_fixed_reaction_rhs_moving_boundary = EQL.EQLModel(ensemble_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, rhs_theta, moving_boundary_theta)
    single_model = EQL.EQLModel(single_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters)
    single_model_fixed_diffusion = EQL.EQLModel(single_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta)
    single_model_fixed_reaction = EQL.EQLModel(single_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta)
    single_model_fixed_rhs = EQL.EQLModel(single_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, rhs_theta)
    single_model_fixed_moving_boundary = EQL.EQLModel(single_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, moving_boundary_theta)
    single_model_fixed_diffusion_reaction = EQL.EQLModel(single_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta)
    single_model_fixed_diffusion_rhs = EQL.EQLModel(single_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, rhs_theta)
    single_model_fixed_diffusion_moving_boundary = EQL.EQLModel(single_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, moving_boundary_theta)
    single_model_fixed_reaction_rhs = EQL.EQLModel(single_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, rhs_theta)
    single_model_fixed_reaction_moving_boundary = EQL.EQLModel(single_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, moving_boundary_theta)
    single_model_fixed_rhs_moving_boundary = EQL.EQLModel(single_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, rhs_theta, moving_boundary_theta)
    single_model_fixed_diffusion_reaction_rhs = EQL.EQLModel(single_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta, rhs_theta)
    single_model_fixed_diffusion_reaction_moving_boundary = EQL.EQLModel(single_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta, moving_boundary_theta)
    single_model_fixed_diffusion_rhs_moving_boundary = EQL.EQLModel(single_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, rhs_theta, moving_boundary_theta)
    single_model_fixed_reaction_rhs_moving_boundary = EQL.EQLModel(single_sol; mesh_points=500, diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, rhs_theta, moving_boundary_theta)

    @test EQL.has_proliferation(ensemble_sol)
    @test EQL.has_proliferation(single_sol)
    @test EQL.get_saveat(ensemble_sol) == ensemble_sol[1].t
    @test EQL.get_saveat(single_sol) == single_sol.t
    Id = fill(true, length(diffusion_theta))
    Ir = fill(true, length(reaction_theta))
    Irhs = fill(true, length(rhs_theta))
    Imb = fill(true, length(moving_boundary_theta))
    @test ensemble_model.indicators == single_model.indicators == [Id; Ir; Irhs; Imb]
    @test ensemble_model_fixed_diffusion.indicators == single_model_fixed_diffusion.indicators == [Ir; Irhs; Imb]
    @test ensemble_model_fixed_reaction.indicators == single_model_fixed_reaction.indicators == [Id; Irhs; Imb]
    @test ensemble_model_fixed_rhs.indicators == single_model_fixed_rhs.indicators == [Id; Ir; Imb]
    @test ensemble_model_fixed_moving_boundary.indicators == single_model_fixed_moving_boundary.indicators == [Id; Ir; Irhs]
    @test ensemble_model_fixed_diffusion_reaction.indicators == single_model_fixed_diffusion_reaction.indicators == [Irhs; Imb]
    @test ensemble_model_fixed_diffusion_rhs.indicators == single_model_fixed_diffusion_rhs.indicators == [Ir; Imb]
    @test ensemble_model_fixed_diffusion_moving_boundary.indicators == single_model_fixed_diffusion_moving_boundary.indicators == [Ir; Irhs]
    @test ensemble_model_fixed_reaction_rhs.indicators == single_model_fixed_reaction_rhs.indicators == [Id; Imb]
    @test ensemble_model_fixed_reaction_moving_boundary.indicators == single_model_fixed_reaction_moving_boundary.indicators == [Id; Irhs]
    @test ensemble_model_fixed_rhs_moving_boundary.indicators == single_model_fixed_rhs_moving_boundary.indicators == [Id; Ir]
    @test ensemble_model_fixed_diffusion_reaction_rhs.indicators == single_model_fixed_diffusion_reaction_rhs.indicators == [Imb;]
    @test ensemble_model_fixed_diffusion_reaction_moving_boundary.indicators == single_model_fixed_diffusion_reaction_moving_boundary.indicators == [Irhs;]
    @test ensemble_model_fixed_diffusion_rhs_moving_boundary.indicators == single_model_fixed_diffusion_rhs_moving_boundary.indicators == [Ir;]
    @test ensemble_model_fixed_reaction_rhs_moving_boundary.indicators == single_model_fixed_reaction_rhs_moving_boundary.indicators == [Id;]

    pde_ensemble_model = EQL.build_pde(ensemble_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters)
    pde_ensemble_model_fixed_diffusion = EQL.build_pde(ensemble_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta)
    pde_ensemble_model_fixed_reaction = EQL.build_pde(ensemble_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta)
    pde_ensemble_model_fixed_rhs = EQL.build_pde(ensemble_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, rhs_theta)
    pde_ensemble_model_fixed_moving_boundary = EQL.build_pde(ensemble_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, moving_boundary_theta)
    pde_ensemble_model_fixed_diffusion_reaction = EQL.build_pde(ensemble_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta)
    pde_ensemble_model_fixed_diffusion_rhs = EQL.build_pde(ensemble_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, rhs_theta)
    pde_ensemble_model_fixed_diffusion_moving_boundary = EQL.build_pde(ensemble_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, moving_boundary_theta)
    pde_ensemble_model_fixed_reaction_rhs = EQL.build_pde(ensemble_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, rhs_theta)
    pde_ensemble_model_fixed_reaction_moving_boundary = EQL.build_pde(ensemble_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, moving_boundary_theta)
    pde_ensemble_model_fixed_rhs_moving_boundary = EQL.build_pde(ensemble_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, rhs_theta, moving_boundary_theta)
    pde_ensemble_model_fixed_diffusion_reaction_rhs = EQL.build_pde(ensemble_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta, rhs_theta)
    pde_ensemble_model_fixed_diffusion_reaction_moving_boundary = EQL.build_pde(ensemble_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta, moving_boundary_theta)
    pde_ensemble_model_fixed_diffusion_rhs_moving_boundary = EQL.build_pde(ensemble_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, rhs_theta, moving_boundary_theta)
    pde_ensemble_model_fixed_reaction_rhs_moving_boundary = EQL.build_pde(ensemble_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, rhs_theta, moving_boundary_theta)
    pde_single_model = EQL.build_pde(single_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters)
    pde_single_model_fixed_diffusion = EQL.build_pde(single_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta)
    pde_single_model_fixed_reaction = EQL.build_pde(single_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta)
    pde_single_model_fixed_rhs = EQL.build_pde(single_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, rhs_theta)
    pde_single_model_fixed_moving_boundary = EQL.build_pde(single_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, moving_boundary_theta)
    pde_single_model_fixed_diffusion_reaction = EQL.build_pde(single_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta)
    pde_single_model_fixed_diffusion_rhs = EQL.build_pde(single_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, rhs_theta)
    pde_single_model_fixed_diffusion_moving_boundary = EQL.build_pde(single_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, moving_boundary_theta)
    pde_single_model_fixed_reaction_rhs = EQL.build_pde(single_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, rhs_theta)
    pde_single_model_fixed_reaction_moving_boundary = EQL.build_pde(single_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, moving_boundary_theta)
    pde_single_model_fixed_rhs_moving_boundary = EQL.build_pde(single_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, rhs_theta, moving_boundary_theta)
    pde_single_model_fixed_diffusion_reaction_rhs = EQL.build_pde(single_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta, rhs_theta)
    pde_single_model_fixed_diffusion_reaction_moving_boundary = EQL.build_pde(single_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, reaction_theta, moving_boundary_theta)
    pde_single_model_fixed_diffusion_rhs_moving_boundary = EQL.build_pde(single_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, diffusion_theta, rhs_theta, moving_boundary_theta)
    pde_single_model_fixed_reaction_rhs_moving_boundary = EQL.build_pde(single_sol, 500; diffusion_basis, diffusion_parameters, reaction_basis, reaction_parameters, rhs_basis, rhs_parameters, moving_boundary_basis, moving_boundary_parameters, reaction_theta, rhs_theta, moving_boundary_theta)
    manual_pde_model = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zero(diffusion_theta), p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zero(rhs_theta), p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zero(moving_boundary_theta), p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=zero(reaction_theta), p=reaction_parameters))
    manual_pde_model_fixed_diffusion = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zero(rhs_theta), p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zero(moving_boundary_theta), p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=zero(reaction_theta), p=reaction_parameters))
    manual_pde_model_fixed_reaction = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zero(diffusion_theta), p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zero(rhs_theta), p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zero(moving_boundary_theta), p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_rhs = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zero(diffusion_theta), p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zero(moving_boundary_theta), p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=zero(reaction_theta), p=reaction_parameters))
    manual_pde_model_fixed_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zero(diffusion_theta), p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zero(rhs_theta), p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=zero(reaction_theta), p=reaction_parameters))
    manual_pde_model_fixed_diffusion_reaction = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zero(rhs_theta), p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zero(moving_boundary_theta), p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_diffusion_rhs = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zero(moving_boundary_theta), p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=zero(reaction_theta), p=reaction_parameters))
    manual_pde_model_fixed_diffusion_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zero(rhs_theta), p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=zero(reaction_theta), p=reaction_parameters))
    manual_pde_model_fixed_reaction_rhs = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zero(diffusion_theta), p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zero(moving_boundary_theta), p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_reaction_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zero(diffusion_theta), p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zero(rhs_theta), p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_rhs_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zero(diffusion_theta), p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=zero(reaction_theta), p=reaction_parameters))
    manual_pde_model_fixed_diffusion_reaction_rhs = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zero(moving_boundary_theta), p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_diffusion_reaction_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zero(rhs_theta), p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_diffusion_rhs_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=zero(reaction_theta), p=reaction_parameters))
    manual_pde_model_fixed_reaction_rhs_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zero(diffusion_theta), p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))

    sol_ensemble_model = solve(pde_ensemble_model, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion = solve(pde_ensemble_model_fixed_diffusion, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_reaction = solve(pde_ensemble_model_fixed_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_rhs = solve(pde_ensemble_model_fixed_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_moving_boundary = solve(pde_ensemble_model_fixed_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_reaction = solve(pde_ensemble_model_fixed_diffusion_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_rhs = solve(pde_ensemble_model_fixed_diffusion_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_moving_boundary = solve(pde_ensemble_model_fixed_diffusion_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_reaction_rhs = solve(pde_ensemble_model_fixed_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_reaction_moving_boundary = solve(pde_ensemble_model_fixed_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_rhs_moving_boundary = solve(pde_ensemble_model_fixed_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_reaction_rhs = solve(pde_ensemble_model_fixed_diffusion_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_reaction_moving_boundary = solve(pde_ensemble_model_fixed_diffusion_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_rhs_moving_boundary = solve(pde_ensemble_model_fixed_diffusion_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_reaction_rhs_moving_boundary = solve(pde_ensemble_model_fixed_reaction_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_single_model = solve(pde_single_model, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion = solve(pde_single_model_fixed_diffusion, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_reaction = solve(pde_single_model_fixed_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_rhs = solve(pde_single_model_fixed_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_moving_boundary = solve(pde_single_model_fixed_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_reaction = solve(pde_single_model_fixed_diffusion_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_rhs = solve(pde_single_model_fixed_diffusion_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_moving_boundary = solve(pde_single_model_fixed_diffusion_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_reaction_rhs = solve(pde_single_model_fixed_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_reaction_moving_boundary = solve(pde_single_model_fixed_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_rhs_moving_boundary = solve(pde_single_model_fixed_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_reaction_rhs = solve(pde_single_model_fixed_diffusion_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_reaction_moving_boundary = solve(pde_single_model_fixed_diffusion_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_rhs_moving_boundary = solve(pde_single_model_fixed_diffusion_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_reaction_rhs_moving_boundary = solve(pde_single_model_fixed_reaction_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_manual_pde_model = solve(manual_pde_model, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion = solve(manual_pde_model_fixed_diffusion, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_reaction = solve(manual_pde_model_fixed_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_rhs = solve(manual_pde_model_fixed_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_moving_boundary = solve(manual_pde_model_fixed_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion_reaction = solve(manual_pde_model_fixed_diffusion_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion_rhs = solve(manual_pde_model_fixed_diffusion_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion_moving_boundary = solve(manual_pde_model_fixed_diffusion_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_reaction_rhs = solve(manual_pde_model_fixed_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_reaction_moving_boundary = solve(manual_pde_model_fixed_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_rhs_moving_boundary = solve(manual_pde_model_fixed_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion_reaction_rhs = solve(manual_pde_model_fixed_diffusion_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion_reaction_moving_boundary = solve(manual_pde_model_fixed_diffusion_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion_rhs_moving_boundary = solve(manual_pde_model_fixed_diffusion_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_reaction_rhs_moving_boundary = solve(manual_pde_model_fixed_reaction_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)

    @test sol_ensemble_model.u == sol_single_model.u == sol_manual_pde_model.u
    @test sol_ensemble_model_fixed_diffusion.u == sol_single_model_fixed_diffusion.u == sol_manual_pde_model_fixed_diffusion.u
    @test sol_ensemble_model_fixed_reaction.u == sol_single_model_fixed_reaction.u == sol_manual_pde_model_fixed_reaction.u
    @test sol_ensemble_model_fixed_rhs.u == sol_single_model_fixed_rhs.u == sol_manual_pde_model_fixed_rhs.u
    @test sol_ensemble_model_fixed_moving_boundary.u == sol_single_model_fixed_moving_boundary.u == sol_manual_pde_model_fixed_moving_boundary.u
    @test sol_ensemble_model_fixed_diffusion_reaction.u == sol_single_model_fixed_diffusion_reaction.u == sol_manual_pde_model_fixed_diffusion_reaction.u
    @test sol_ensemble_model_fixed_diffusion_rhs.u == sol_single_model_fixed_diffusion_rhs.u == sol_manual_pde_model_fixed_diffusion_rhs.u
    @test sol_ensemble_model_fixed_diffusion_moving_boundary.u == sol_single_model_fixed_diffusion_moving_boundary.u == sol_manual_pde_model_fixed_diffusion_moving_boundary.u
    @test sol_ensemble_model_fixed_reaction_rhs.u == sol_single_model_fixed_reaction_rhs.u == sol_manual_pde_model_fixed_reaction_rhs.u
    @test sol_ensemble_model_fixed_reaction_moving_boundary.u == sol_single_model_fixed_reaction_moving_boundary.u == sol_manual_pde_model_fixed_reaction_moving_boundary.u
    @test sol_ensemble_model_fixed_rhs_moving_boundary.u == sol_single_model_fixed_rhs_moving_boundary.u == sol_manual_pde_model_fixed_rhs_moving_boundary.u
    @test sol_ensemble_model_fixed_diffusion_reaction_rhs.u == sol_single_model_fixed_diffusion_reaction_rhs.u == sol_manual_pde_model_fixed_diffusion_reaction_rhs.u
    @test sol_ensemble_model_fixed_diffusion_reaction_moving_boundary.u == sol_single_model_fixed_diffusion_reaction_moving_boundary.u == sol_manual_pde_model_fixed_diffusion_reaction_moving_boundary.u
    @test sol_ensemble_model_fixed_diffusion_rhs_moving_boundary.u == sol_single_model_fixed_diffusion_rhs_moving_boundary.u == sol_manual_pde_model_fixed_diffusion_rhs_moving_boundary.u
    @test sol_ensemble_model_fixed_reaction_rhs_moving_boundary.u == sol_single_model_fixed_reaction_rhs_moving_boundary.u == sol_manual_pde_model_fixed_reaction_rhs_moving_boundary.u

    # Rebuilding: Zeros 
    pde_ensemble_model = EQL.rebuild_pde(ensemble_model)
    pde_ensemble_model_fixed_diffusion = EQL.rebuild_pde(ensemble_model_fixed_diffusion)
    pde_ensemble_model_fixed_reaction = EQL.rebuild_pde(ensemble_model_fixed_reaction)
    pde_ensemble_model_fixed_rhs = EQL.rebuild_pde(ensemble_model_fixed_rhs)
    pde_ensemble_model_fixed_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_moving_boundary)
    pde_ensemble_model_fixed_diffusion_reaction = EQL.rebuild_pde(ensemble_model_fixed_diffusion_reaction)
    pde_ensemble_model_fixed_diffusion_rhs = EQL.rebuild_pde(ensemble_model_fixed_diffusion_rhs)
    pde_ensemble_model_fixed_diffusion_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_diffusion_moving_boundary)
    pde_ensemble_model_fixed_reaction_rhs = EQL.rebuild_pde(ensemble_model_fixed_reaction_rhs)
    pde_ensemble_model_fixed_reaction_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_reaction_moving_boundary)
    pde_ensemble_model_fixed_rhs_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_rhs_moving_boundary)
    pde_ensemble_model_fixed_diffusion_reaction_rhs = EQL.rebuild_pde(ensemble_model_fixed_diffusion_reaction_rhs)
    pde_ensemble_model_fixed_diffusion_reaction_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_diffusion_reaction_moving_boundary)
    pde_ensemble_model_fixed_diffusion_rhs_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_diffusion_rhs_moving_boundary)
    pde_ensemble_model_fixed_reaction_rhs_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_reaction_rhs_moving_boundary)
    pde_single_model = EQL.rebuild_pde(single_model)
    pde_single_model_fixed_diffusion = EQL.rebuild_pde(single_model_fixed_diffusion)
    pde_single_model_fixed_reaction = EQL.rebuild_pde(single_model_fixed_reaction)
    pde_single_model_fixed_rhs = EQL.rebuild_pde(single_model_fixed_rhs)
    pde_single_model_fixed_moving_boundary = EQL.rebuild_pde(single_model_fixed_moving_boundary)
    pde_single_model_fixed_diffusion_reaction = EQL.rebuild_pde(single_model_fixed_diffusion_reaction)
    pde_single_model_fixed_diffusion_rhs = EQL.rebuild_pde(single_model_fixed_diffusion_rhs)
    pde_single_model_fixed_diffusion_moving_boundary = EQL.rebuild_pde(single_model_fixed_diffusion_moving_boundary)
    pde_single_model_fixed_reaction_rhs = EQL.rebuild_pde(single_model_fixed_reaction_rhs)
    pde_single_model_fixed_reaction_moving_boundary = EQL.rebuild_pde(single_model_fixed_reaction_moving_boundary)
    pde_single_model_fixed_rhs_moving_boundary = EQL.rebuild_pde(single_model_fixed_rhs_moving_boundary)
    pde_single_model_fixed_diffusion_reaction_rhs = EQL.rebuild_pde(single_model_fixed_diffusion_reaction_rhs)
    pde_single_model_fixed_diffusion_reaction_moving_boundary = EQL.rebuild_pde(single_model_fixed_diffusion_reaction_moving_boundary)
    pde_single_model_fixed_diffusion_rhs_moving_boundary = EQL.rebuild_pde(single_model_fixed_diffusion_rhs_moving_boundary)
    pde_single_model_fixed_reaction_rhs_moving_boundary = EQL.rebuild_pde(single_model_fixed_reaction_rhs_moving_boundary)
    sol_ensemble_model = solve(pde_ensemble_model, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion = solve(pde_ensemble_model_fixed_diffusion, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_reaction = solve(pde_ensemble_model_fixed_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_rhs = solve(pde_ensemble_model_fixed_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_moving_boundary = solve(pde_ensemble_model_fixed_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_reaction = solve(pde_ensemble_model_fixed_diffusion_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_rhs = solve(pde_ensemble_model_fixed_diffusion_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_moving_boundary = solve(pde_ensemble_model_fixed_diffusion_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_reaction_rhs = solve(pde_ensemble_model_fixed_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_reaction_moving_boundary = solve(pde_ensemble_model_fixed_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_rhs_moving_boundary = solve(pde_ensemble_model_fixed_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_reaction_rhs = solve(pde_ensemble_model_fixed_diffusion_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_reaction_moving_boundary = solve(pde_ensemble_model_fixed_diffusion_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_rhs_moving_boundary = solve(pde_ensemble_model_fixed_diffusion_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_reaction_rhs_moving_boundary = solve(pde_ensemble_model_fixed_reaction_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_single_model = solve(pde_single_model, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion = solve(pde_single_model_fixed_diffusion, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_reaction = solve(pde_single_model_fixed_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_rhs = solve(pde_single_model_fixed_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_moving_boundary = solve(pde_single_model_fixed_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_reaction = solve(pde_single_model_fixed_diffusion_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_rhs = solve(pde_single_model_fixed_diffusion_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_moving_boundary = solve(pde_single_model_fixed_diffusion_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_reaction_rhs = solve(pde_single_model_fixed_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_reaction_moving_boundary = solve(pde_single_model_fixed_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_rhs_moving_boundary = solve(pde_single_model_fixed_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_reaction_rhs = solve(pde_single_model_fixed_diffusion_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_reaction_moving_boundary = solve(pde_single_model_fixed_diffusion_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_rhs_moving_boundary = solve(pde_single_model_fixed_diffusion_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_reaction_rhs_moving_boundary = solve(pde_single_model_fixed_reaction_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    @test sol_ensemble_model.u == sol_single_model.u == sol_manual_pde_model.u
    @test sol_ensemble_model_fixed_diffusion.u == sol_single_model_fixed_diffusion.u == sol_manual_pde_model_fixed_diffusion.u
    @test sol_ensemble_model_fixed_reaction.u == sol_single_model_fixed_reaction.u == sol_manual_pde_model_fixed_reaction.u
    @test sol_ensemble_model_fixed_rhs.u == sol_single_model_fixed_rhs.u == sol_manual_pde_model_fixed_rhs.u
    @test sol_ensemble_model_fixed_moving_boundary.u == sol_single_model_fixed_moving_boundary.u == sol_manual_pde_model_fixed_moving_boundary.u
    @test sol_ensemble_model_fixed_diffusion_reaction.u == sol_single_model_fixed_diffusion_reaction.u == sol_manual_pde_model_fixed_diffusion_reaction.u
    @test sol_ensemble_model_fixed_diffusion_rhs.u == sol_single_model_fixed_diffusion_rhs.u == sol_manual_pde_model_fixed_diffusion_rhs.u
    @test sol_ensemble_model_fixed_diffusion_moving_boundary.u == sol_single_model_fixed_diffusion_moving_boundary.u == sol_manual_pde_model_fixed_diffusion_moving_boundary.u
    @test sol_ensemble_model_fixed_reaction_rhs.u == sol_single_model_fixed_reaction_rhs.u == sol_manual_pde_model_fixed_reaction_rhs.u
    @test sol_ensemble_model_fixed_reaction_moving_boundary.u == sol_single_model_fixed_reaction_moving_boundary.u == sol_manual_pde_model_fixed_reaction_moving_boundary.u
    @test sol_ensemble_model_fixed_rhs_moving_boundary.u == sol_single_model_fixed_rhs_moving_boundary.u == sol_manual_pde_model_fixed_rhs_moving_boundary.u
    @test sol_ensemble_model_fixed_diffusion_reaction_rhs.u == sol_single_model_fixed_diffusion_reaction_rhs.u == sol_manual_pde_model_fixed_diffusion_reaction_rhs.u
    @test sol_ensemble_model_fixed_diffusion_reaction_moving_boundary.u == sol_single_model_fixed_diffusion_reaction_moving_boundary.u == sol_manual_pde_model_fixed_diffusion_reaction_moving_boundary.u
    @test sol_ensemble_model_fixed_diffusion_rhs_moving_boundary.u == sol_single_model_fixed_diffusion_rhs_moving_boundary.u == sol_manual_pde_model_fixed_diffusion_rhs_moving_boundary.u
    @test sol_ensemble_model_fixed_reaction_rhs_moving_boundary.u == sol_single_model_fixed_reaction_rhs_moving_boundary.u == sol_manual_pde_model_fixed_reaction_rhs_moving_boundary.u

    # Rebuilding: A new vector 
    θd = diffusion_theta + 1e-2randn(length(diffusion_theta))
    θr = reaction_theta + 1e-2randn(length(reaction_theta))
    θrhs = rhs_theta + 1e-2randn(length(rhs_theta))
    θmb = moving_boundary_theta + 1e-2randn(length(moving_boundary_theta))
    pde_ensemble_model = EQL.rebuild_pde(ensemble_model, [θd; θr; θrhs; θmb])
    pde_ensemble_model_fixed_diffusion = EQL.rebuild_pde(ensemble_model_fixed_diffusion, [θr; θrhs; θmb])
    pde_ensemble_model_fixed_reaction = EQL.rebuild_pde(ensemble_model_fixed_reaction, [θd; θrhs; θmb])
    pde_ensemble_model_fixed_rhs = EQL.rebuild_pde(ensemble_model_fixed_rhs, [θd; θr; θmb])
    pde_ensemble_model_fixed_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_moving_boundary, [θd; θr; θrhs])
    pde_ensemble_model_fixed_diffusion_reaction = EQL.rebuild_pde(ensemble_model_fixed_diffusion_reaction, [θrhs; θmb])
    pde_ensemble_model_fixed_diffusion_rhs = EQL.rebuild_pde(ensemble_model_fixed_diffusion_rhs, [θr; θmb])
    pde_ensemble_model_fixed_diffusion_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_diffusion_moving_boundary, [θr; θrhs])
    pde_ensemble_model_fixed_reaction_rhs = EQL.rebuild_pde(ensemble_model_fixed_reaction_rhs, [θd; θmb])
    pde_ensemble_model_fixed_reaction_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_reaction_moving_boundary, [θd; θrhs])
    pde_ensemble_model_fixed_rhs_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_rhs_moving_boundary, [θd; θr])
    pde_ensemble_model_fixed_diffusion_reaction_rhs = EQL.rebuild_pde(ensemble_model_fixed_diffusion_reaction_rhs, [θmb;])
    pde_ensemble_model_fixed_diffusion_reaction_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_diffusion_reaction_moving_boundary, [θrhs;])
    pde_ensemble_model_fixed_diffusion_rhs_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_diffusion_rhs_moving_boundary, [θr;])
    pde_ensemble_model_fixed_reaction_rhs_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_reaction_rhs_moving_boundary, [θd;])
    pde_single_model = EQL.rebuild_pde(single_model, [θd; θr; θrhs; θmb])
    pde_single_model_fixed_diffusion = EQL.rebuild_pde(single_model_fixed_diffusion, [θr; θrhs; θmb])
    pde_single_model_fixed_reaction = EQL.rebuild_pde(single_model_fixed_reaction, [θd; θrhs; θmb])
    pde_single_model_fixed_rhs = EQL.rebuild_pde(single_model_fixed_rhs, [θd; θr; θmb])
    pde_single_model_fixed_moving_boundary = EQL.rebuild_pde(single_model_fixed_moving_boundary, [θd; θr; θrhs])
    pde_single_model_fixed_diffusion_reaction = EQL.rebuild_pde(single_model_fixed_diffusion_reaction, [θrhs; θmb])
    pde_single_model_fixed_diffusion_rhs = EQL.rebuild_pde(single_model_fixed_diffusion_rhs, [θr; θmb])
    pde_single_model_fixed_diffusion_moving_boundary = EQL.rebuild_pde(single_model_fixed_diffusion_moving_boundary, [θr; θrhs])
    pde_single_model_fixed_reaction_rhs = EQL.rebuild_pde(single_model_fixed_reaction_rhs, [θd; θmb])
    pde_single_model_fixed_reaction_moving_boundary = EQL.rebuild_pde(single_model_fixed_reaction_moving_boundary, [θd; θrhs])
    pde_single_model_fixed_rhs_moving_boundary = EQL.rebuild_pde(single_model_fixed_rhs_moving_boundary, [θd; θr])
    pde_single_model_fixed_diffusion_reaction_rhs = EQL.rebuild_pde(single_model_fixed_diffusion_reaction_rhs, [θmb;])
    pde_single_model_fixed_diffusion_reaction_moving_boundary = EQL.rebuild_pde(single_model_fixed_diffusion_reaction_moving_boundary, [θrhs;])
    pde_single_model_fixed_diffusion_rhs_moving_boundary = EQL.rebuild_pde(single_model_fixed_diffusion_rhs_moving_boundary, [θr;])
    pde_single_model_fixed_reaction_rhs_moving_boundary = EQL.rebuild_pde(single_model_fixed_reaction_rhs_moving_boundary, [θd;])
    manual_pde_model = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=θd, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=θrhs, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=θmb, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=θr, p=reaction_parameters))
    manual_pde_model_fixed_diffusion = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=θrhs, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=θmb, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=θr, p=reaction_parameters))
    manual_pde_model_fixed_reaction = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=θd, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=θrhs, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=θmb, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_rhs = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=θd, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=θmb, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=θr, p=reaction_parameters))
    manual_pde_model_fixed_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=θd, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=θrhs, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=θr, p=reaction_parameters))
    manual_pde_model_fixed_diffusion_reaction = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=θrhs, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=θmb, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_diffusion_rhs = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=θmb, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=θr, p=reaction_parameters))
    manual_pde_model_fixed_diffusion_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=θrhs, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=θr, p=reaction_parameters))
    manual_pde_model_fixed_reaction_rhs = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=θd, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=θmb, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_reaction_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=θd, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=θrhs, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_rhs_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=θd, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=θr, p=reaction_parameters))
    manual_pde_model_fixed_diffusion_reaction_rhs = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=θmb, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_diffusion_reaction_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=θrhs, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_diffusion_rhs_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=θr, p=reaction_parameters))
    manual_pde_model_fixed_reaction_rhs_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=θd, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    sol_ensemble_model = solve(pde_ensemble_model, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion = solve(pde_ensemble_model_fixed_diffusion, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_reaction = solve(pde_ensemble_model_fixed_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_rhs = solve(pde_ensemble_model_fixed_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_moving_boundary = solve(pde_ensemble_model_fixed_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_reaction = solve(pde_ensemble_model_fixed_diffusion_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_rhs = solve(pde_ensemble_model_fixed_diffusion_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_moving_boundary = solve(pde_ensemble_model_fixed_diffusion_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_reaction_rhs = solve(pde_ensemble_model_fixed_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_reaction_moving_boundary = solve(pde_ensemble_model_fixed_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_rhs_moving_boundary = solve(pde_ensemble_model_fixed_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_reaction_rhs = solve(pde_ensemble_model_fixed_diffusion_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_reaction_moving_boundary = solve(pde_ensemble_model_fixed_diffusion_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_rhs_moving_boundary = solve(pde_ensemble_model_fixed_diffusion_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_reaction_rhs_moving_boundary = solve(pde_ensemble_model_fixed_reaction_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_single_model = solve(pde_single_model, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion = solve(pde_single_model_fixed_diffusion, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_reaction = solve(pde_single_model_fixed_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_rhs = solve(pde_single_model_fixed_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_moving_boundary = solve(pde_single_model_fixed_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_reaction = solve(pde_single_model_fixed_diffusion_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_rhs = solve(pde_single_model_fixed_diffusion_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_moving_boundary = solve(pde_single_model_fixed_diffusion_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_reaction_rhs = solve(pde_single_model_fixed_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_reaction_moving_boundary = solve(pde_single_model_fixed_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_rhs_moving_boundary = solve(pde_single_model_fixed_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_reaction_rhs = solve(pde_single_model_fixed_diffusion_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_reaction_moving_boundary = solve(pde_single_model_fixed_diffusion_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_rhs_moving_boundary = solve(pde_single_model_fixed_diffusion_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_reaction_rhs_moving_boundary = solve(pde_single_model_fixed_reaction_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_manual_pde_model = solve(manual_pde_model, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion = solve(manual_pde_model_fixed_diffusion, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_reaction = solve(manual_pde_model_fixed_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_rhs = solve(manual_pde_model_fixed_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_moving_boundary = solve(manual_pde_model_fixed_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion_reaction = solve(manual_pde_model_fixed_diffusion_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion_rhs = solve(manual_pde_model_fixed_diffusion_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion_moving_boundary = solve(manual_pde_model_fixed_diffusion_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_reaction_rhs = solve(manual_pde_model_fixed_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_reaction_moving_boundary = solve(manual_pde_model_fixed_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_rhs_moving_boundary = solve(manual_pde_model_fixed_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion_reaction_rhs = solve(manual_pde_model_fixed_diffusion_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion_reaction_moving_boundary = solve(manual_pde_model_fixed_diffusion_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion_rhs_moving_boundary = solve(manual_pde_model_fixed_diffusion_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_reaction_rhs_moving_boundary = solve(manual_pde_model_fixed_reaction_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    @test sol_ensemble_model.u == sol_single_model.u == sol_manual_pde_model.u
    @test sol_ensemble_model_fixed_diffusion.u == sol_single_model_fixed_diffusion.u == sol_manual_pde_model_fixed_diffusion.u
    @test sol_ensemble_model_fixed_reaction.u == sol_single_model_fixed_reaction.u == sol_manual_pde_model_fixed_reaction.u
    @test sol_ensemble_model_fixed_rhs.u == sol_single_model_fixed_rhs.u == sol_manual_pde_model_fixed_rhs.u
    @test sol_ensemble_model_fixed_moving_boundary.u == sol_single_model_fixed_moving_boundary.u == sol_manual_pde_model_fixed_moving_boundary.u
    @test sol_ensemble_model_fixed_diffusion_reaction.u == sol_single_model_fixed_diffusion_reaction.u == sol_manual_pde_model_fixed_diffusion_reaction.u
    @test sol_ensemble_model_fixed_diffusion_rhs.u == sol_single_model_fixed_diffusion_rhs.u == sol_manual_pde_model_fixed_diffusion_rhs.u
    @test sol_ensemble_model_fixed_diffusion_moving_boundary.u == sol_single_model_fixed_diffusion_moving_boundary.u == sol_manual_pde_model_fixed_diffusion_moving_boundary.u
    @test sol_ensemble_model_fixed_reaction_rhs.u == sol_single_model_fixed_reaction_rhs.u == sol_manual_pde_model_fixed_reaction_rhs.u
    @test sol_ensemble_model_fixed_reaction_moving_boundary.u == sol_single_model_fixed_reaction_moving_boundary.u == sol_manual_pde_model_fixed_reaction_moving_boundary.u
    @test sol_ensemble_model_fixed_rhs_moving_boundary.u == sol_single_model_fixed_rhs_moving_boundary.u == sol_manual_pde_model_fixed_rhs_moving_boundary.u
    @test sol_ensemble_model_fixed_diffusion_reaction_rhs.u == sol_single_model_fixed_diffusion_reaction_rhs.u == sol_manual_pde_model_fixed_diffusion_reaction_rhs.u
    @test sol_ensemble_model_fixed_diffusion_reaction_moving_boundary.u == sol_single_model_fixed_diffusion_reaction_moving_boundary.u == sol_manual_pde_model_fixed_diffusion_reaction_moving_boundary.u
    @test sol_ensemble_model_fixed_diffusion_rhs_moving_boundary.u == sol_single_model_fixed_diffusion_rhs_moving_boundary.u == sol_manual_pde_model_fixed_diffusion_rhs_moving_boundary.u
    @test sol_ensemble_model_fixed_reaction_rhs_moving_boundary.u == sol_single_model_fixed_reaction_rhs_moving_boundary.u == sol_manual_pde_model_fixed_reaction_rhs_moving_boundary.u

    # Rebuilding: Partial model 
    θd = diffusion_theta + 1e-12randn(length(diffusion_theta))
    θr = reaction_theta + 1e-12randn(length(reaction_theta))
    θrhs = rhs_theta + 1e-12randn(length(rhs_theta))
    θmb = moving_boundary_theta + 1e-12randn(length(moving_boundary_theta))
    Id = [false, true, false]
    Ir = [true, true, false]
    Irhs = [true, true, false, false]
    Imb = [false, true, false]
    zdiffusion_theta = copy(θd)
    zdiffusion_theta[.!Id] .= 0.0
    zreaction_theta = copy(θr)
    zreaction_theta[.!Ir] .= 0.0
    zrhs_theta = copy(θrhs)
    zrhs_theta[.!Irhs] .= 0.0
    zmoving_boundary_theta = copy(θmb)
    zmoving_boundary_theta[.!Imb] .= 0.0
    pde_ensemble_model = EQL.rebuild_pde(ensemble_model, [θd; θr; θrhs; θmb], [Id; Ir; Irhs; Imb])
    pde_ensemble_model_fixed_diffusion = EQL.rebuild_pde(ensemble_model_fixed_diffusion, [θr; θrhs; θmb], [Ir; Irhs; Imb])
    pde_ensemble_model_fixed_reaction = EQL.rebuild_pde(ensemble_model_fixed_reaction, [θd; θrhs; θmb], [Id; Irhs; Imb])
    pde_ensemble_model_fixed_rhs = EQL.rebuild_pde(ensemble_model_fixed_rhs, [θd; θr; θmb], [Id; Ir; Imb])
    pde_ensemble_model_fixed_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_moving_boundary, [θd; θr; θrhs], [Id; Ir; Irhs])
    pde_ensemble_model_fixed_diffusion_reaction = EQL.rebuild_pde(ensemble_model_fixed_diffusion_reaction, [θrhs; θmb], [Irhs; Imb])
    pde_ensemble_model_fixed_diffusion_rhs = EQL.rebuild_pde(ensemble_model_fixed_diffusion_rhs, [θr; θmb], [Ir; Imb])
    pde_ensemble_model_fixed_diffusion_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_diffusion_moving_boundary, [θr; θrhs], [Ir; Irhs])
    pde_ensemble_model_fixed_reaction_rhs = EQL.rebuild_pde(ensemble_model_fixed_reaction_rhs, [θd; θmb], [Id; Imb])
    pde_ensemble_model_fixed_reaction_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_reaction_moving_boundary, [θd; θrhs], [Id; Irhs])
    pde_ensemble_model_fixed_rhs_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_rhs_moving_boundary, [θd; θr], [Id; Ir])
    pde_ensemble_model_fixed_diffusion_reaction_rhs = EQL.rebuild_pde(ensemble_model_fixed_diffusion_reaction_rhs, [θmb;], [Imb;])
    pde_ensemble_model_fixed_diffusion_reaction_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_diffusion_reaction_moving_boundary, [θrhs;], [Irhs;])
    pde_ensemble_model_fixed_diffusion_rhs_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_diffusion_rhs_moving_boundary, [θr;], [Ir;])
    pde_ensemble_model_fixed_reaction_rhs_moving_boundary = EQL.rebuild_pde(ensemble_model_fixed_reaction_rhs_moving_boundary, [θd;], [Id;])
    pde_single_model = EQL.rebuild_pde(single_model, [θd; θr; θrhs; θmb], [Id; Ir; Irhs; Imb])
    pde_single_model_fixed_diffusion = EQL.rebuild_pde(single_model_fixed_diffusion, [θr; θrhs; θmb], [Ir; Irhs; Imb])
    pde_single_model_fixed_reaction = EQL.rebuild_pde(single_model_fixed_reaction, [θd; θrhs; θmb], [Id; Irhs; Imb])
    pde_single_model_fixed_rhs = EQL.rebuild_pde(single_model_fixed_rhs, [θd; θr; θmb], [Id; Ir; Imb])
    pde_single_model_fixed_moving_boundary = EQL.rebuild_pde(single_model_fixed_moving_boundary, [θd; θr; θrhs], [Id; Ir; Irhs])
    pde_single_model_fixed_diffusion_reaction = EQL.rebuild_pde(single_model_fixed_diffusion_reaction, [θrhs; θmb], [Irhs; Imb])
    pde_single_model_fixed_diffusion_rhs = EQL.rebuild_pde(single_model_fixed_diffusion_rhs, [θr; θmb], [Ir; Imb])
    pde_single_model_fixed_diffusion_moving_boundary = EQL.rebuild_pde(single_model_fixed_diffusion_moving_boundary, [θr; θrhs], [Ir; Irhs])
    pde_single_model_fixed_reaction_rhs = EQL.rebuild_pde(single_model_fixed_reaction_rhs, [θd; θmb], [Id; Imb])
    pde_single_model_fixed_reaction_moving_boundary = EQL.rebuild_pde(single_model_fixed_reaction_moving_boundary, [θd; θrhs], [Id; Irhs])
    pde_single_model_fixed_rhs_moving_boundary = EQL.rebuild_pde(single_model_fixed_rhs_moving_boundary, [θd; θr], [Id; Ir])
    pde_single_model_fixed_diffusion_reaction_rhs = EQL.rebuild_pde(single_model_fixed_diffusion_reaction_rhs, [θmb;], [Imb;])
    pde_single_model_fixed_diffusion_reaction_moving_boundary = EQL.rebuild_pde(single_model_fixed_diffusion_reaction_moving_boundary, [θrhs;], [Irhs;])
    pde_single_model_fixed_diffusion_rhs_moving_boundary = EQL.rebuild_pde(single_model_fixed_diffusion_rhs_moving_boundary, [θr;], [Ir;])
    pde_single_model_fixed_reaction_rhs_moving_boundary = EQL.rebuild_pde(single_model_fixed_reaction_rhs_moving_boundary, [θd;], [Id;])
    manual_pde_model = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zdiffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zrhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zmoving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=zreaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_diffusion = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zrhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zmoving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=zreaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_reaction = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zdiffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zrhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zmoving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_rhs = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zdiffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zmoving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=zreaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zdiffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zrhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=zreaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_diffusion_reaction = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zrhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zmoving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_diffusion_rhs = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zmoving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=zreaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_diffusion_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zrhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=zreaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_reaction_rhs = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zdiffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zmoving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_reaction_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zdiffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zrhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_rhs_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zdiffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=zreaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_diffusion_reaction_rhs = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=zmoving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_diffusion_reaction_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=zrhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_diffusion_rhs_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=diffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=zreaction_theta, p=reaction_parameters))
    manual_pde_model_fixed_reaction_rhs_moving_boundary = MBProblem(prob, 500; diffusion_function=diffusion_basis, diffusion_parameters=EQL.Parameters(θ=zdiffusion_theta, p=diffusion_parameters), rhs_function=rhs_basis, rhs_parameters=EQL.Parameters(θ=rhs_theta, p=rhs_parameters), moving_boundary_function=(u, t, p) -> (zero(u), -inv(u)*moving_boundary_basis(u, t, p)), moving_boundary_parameters=EQL.Parameters(θ=moving_boundary_theta, p=moving_boundary_parameters), proliferation=true, reaction_function=reaction_basis, reaction_parameters=EQL.Parameters(θ=reaction_theta, p=reaction_parameters))
    sol_ensemble_model = solve(pde_ensemble_model, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion = solve(pde_ensemble_model_fixed_diffusion, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_reaction = solve(pde_ensemble_model_fixed_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_rhs = solve(pde_ensemble_model_fixed_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_moving_boundary = solve(pde_ensemble_model_fixed_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_reaction = solve(pde_ensemble_model_fixed_diffusion_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_rhs = solve(pde_ensemble_model_fixed_diffusion_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_moving_boundary = solve(pde_ensemble_model_fixed_diffusion_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_reaction_rhs = solve(pde_ensemble_model_fixed_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_reaction_moving_boundary = solve(pde_ensemble_model_fixed_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_rhs_moving_boundary = solve(pde_ensemble_model_fixed_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_reaction_rhs = solve(pde_ensemble_model_fixed_diffusion_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_reaction_moving_boundary = solve(pde_ensemble_model_fixed_diffusion_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_diffusion_rhs_moving_boundary = solve(pde_ensemble_model_fixed_diffusion_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_ensemble_model_fixed_reaction_rhs_moving_boundary = solve(pde_ensemble_model_fixed_reaction_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_single_model = solve(pde_single_model, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion = solve(pde_single_model_fixed_diffusion, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_reaction = solve(pde_single_model_fixed_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_rhs = solve(pde_single_model_fixed_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_moving_boundary = solve(pde_single_model_fixed_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_reaction = solve(pde_single_model_fixed_diffusion_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_rhs = solve(pde_single_model_fixed_diffusion_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_moving_boundary = solve(pde_single_model_fixed_diffusion_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_reaction_rhs = solve(pde_single_model_fixed_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_reaction_moving_boundary = solve(pde_single_model_fixed_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_rhs_moving_boundary = solve(pde_single_model_fixed_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_reaction_rhs = solve(pde_single_model_fixed_diffusion_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_reaction_moving_boundary = solve(pde_single_model_fixed_diffusion_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_diffusion_rhs_moving_boundary = solve(pde_single_model_fixed_diffusion_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_single_model_fixed_reaction_rhs_moving_boundary = solve(pde_single_model_fixed_reaction_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=single_sol.t)
    sol_manual_pde_model = solve(manual_pde_model, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion = solve(manual_pde_model_fixed_diffusion, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_reaction = solve(manual_pde_model_fixed_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_rhs = solve(manual_pde_model_fixed_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_moving_boundary = solve(manual_pde_model_fixed_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion_reaction = solve(manual_pde_model_fixed_diffusion_reaction, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion_rhs = solve(manual_pde_model_fixed_diffusion_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion_moving_boundary = solve(manual_pde_model_fixed_diffusion_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_reaction_rhs = solve(manual_pde_model_fixed_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_reaction_moving_boundary = solve(manual_pde_model_fixed_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_rhs_moving_boundary = solve(manual_pde_model_fixed_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion_reaction_rhs = solve(manual_pde_model_fixed_diffusion_reaction_rhs, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion_reaction_moving_boundary = solve(manual_pde_model_fixed_diffusion_reaction_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_diffusion_rhs_moving_boundary = solve(manual_pde_model_fixed_diffusion_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    sol_manual_pde_model_fixed_reaction_rhs_moving_boundary = solve(manual_pde_model_fixed_reaction_rhs_moving_boundary, TRBDF2(linsolve=KLUFactorization()), saveat=ensemble_sol[1].t)
    @test sol_ensemble_model.u == sol_single_model.u == sol_manual_pde_model.u
    @test sol_ensemble_model_fixed_diffusion.u == sol_single_model_fixed_diffusion.u == sol_manual_pde_model_fixed_diffusion.u
    @test sol_ensemble_model_fixed_reaction.u == sol_single_model_fixed_reaction.u == sol_manual_pde_model_fixed_reaction.u
    @test sol_ensemble_model_fixed_rhs.u == sol_single_model_fixed_rhs.u == sol_manual_pde_model_fixed_rhs.u
    @test sol_ensemble_model_fixed_moving_boundary.u == sol_single_model_fixed_moving_boundary.u == sol_manual_pde_model_fixed_moving_boundary.u
    @test sol_ensemble_model_fixed_diffusion_reaction.u == sol_single_model_fixed_diffusion_reaction.u == sol_manual_pde_model_fixed_diffusion_reaction.u
    @test sol_ensemble_model_fixed_diffusion_rhs.u == sol_single_model_fixed_diffusion_rhs.u == sol_manual_pde_model_fixed_diffusion_rhs.u
    @test sol_ensemble_model_fixed_diffusion_moving_boundary.u == sol_single_model_fixed_diffusion_moving_boundary.u == sol_manual_pde_model_fixed_diffusion_moving_boundary.u
    @test sol_ensemble_model_fixed_reaction_rhs.u == sol_single_model_fixed_reaction_rhs.u == sol_manual_pde_model_fixed_reaction_rhs.u
    @test sol_ensemble_model_fixed_reaction_moving_boundary.u == sol_single_model_fixed_reaction_moving_boundary.u == sol_manual_pde_model_fixed_reaction_moving_boundary.u
    @test sol_ensemble_model_fixed_rhs_moving_boundary.u == sol_single_model_fixed_rhs_moving_boundary.u == sol_manual_pde_model_fixed_rhs_moving_boundary.u
    @test sol_ensemble_model_fixed_diffusion_reaction_rhs.u == sol_single_model_fixed_diffusion_reaction_rhs.u == sol_manual_pde_model_fixed_diffusion_reaction_rhs.u
    @test sol_ensemble_model_fixed_diffusion_reaction_moving_boundary.u == sol_single_model_fixed_diffusion_reaction_moving_boundary.u == sol_manual_pde_model_fixed_diffusion_reaction_moving_boundary.u
    @test sol_ensemble_model_fixed_diffusion_rhs_moving_boundary.u == sol_single_model_fixed_diffusion_rhs_moving_boundary.u == sol_manual_pde_model_fixed_diffusion_rhs_moving_boundary.u
    @test sol_ensemble_model_fixed_reaction_rhs_moving_boundary.u == sol_single_model_fixed_reaction_rhs_moving_boundary.u == sol_manual_pde_model_fixed_reaction_rhs_moving_boundary.u
end