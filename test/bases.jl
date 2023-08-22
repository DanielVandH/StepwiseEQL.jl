using ..StepwiseEQL
const EQL = StepwiseEQL

@testset "Parameters" begin
    pθ1 = EQL.Parameters(; θ=nothing)
    pθ2 = EQL.Parameters(-0.1, 0.5)
    @test pθ1.p === nothing && pθ1.θ === nothing
    @test pθ2.p == 0.5 && pθ2.θ == -0.1
end

@testset "BasisSet" begin
    ϕ1 = (u, p) -> u
    ϕ2 = (u, p) -> u^2
    ϕ3 = (u, p) -> u^3
    bases = BasisSet(ϕ1, ϕ2, ϕ3)
    @test bases.bases == (ϕ1, ϕ2, ϕ3)
    @test length(bases) == 3
    @test eachindex(bases) == 1:3
    bas = BasisSet()
    @test bas.bases == ()
    @test length(bas) == 0
    @test eachindex(bas) == 1:0
    @test bas(0.5,[0.5],0.5) == 0.0
    @inferred bas(0.5,[0.5],0.5)
end

@testset "PolynomialBasis" begin
    ϕ = PolynomialBasis(1, 3)
    @test all(d -> 0.7^d == ϕ.bases[d](0.7, nothing), 1:3)
    ϕ = PolynomialBasis(2, 8)
    @test all(d -> 0.7^d == ϕ.bases[d-1](0.7, nothing), 2:8)
    ϕ = PolynomialBasis(-4, 7)
    @test all(d -> 0.7^d == ϕ.bases[d+5](0.7, nothing), -4:7)
    @inferred ϕ.bases[1](0.7, nothing)
    @test length(ϕ) == length(ϕ.bases)
    ϕ = PolynomialBasis(-1, -3)
    @test length(ϕ) == length(ϕ.bases)
    @test ϕ.bases == (ϕ.bases[1], ϕ.bases[2], ϕ.bases[3])
    @inferred ϕ.bases[1](0.7, nothing)
    @test all(d -> 0.7^d == ϕ.bases[d+1](0.7, nothing), -1:-3)
end

@testset "Evaluating Bases" begin
    bases = PolynomialBasis(-3, -1)
    p = nothing
    θ = [1.7, 0.5, -1.3]
    @test bases(0.7, θ, p) == 1.7 * 0.7^-3 + 0.5 * 0.7^-2 + -1.3 * 0.7^-1
    @inferred bases(0.7, θ, p)
    @test bases(0.7, rand(), rand(), EQL.Parameters(θ, p)) == bases(0.7, θ, p)
    @inferred bases(0.7, rand(), rand(), EQL.Parameters(θ, p))
    @test bases(0.7, rand(), EQL.Parameters(θ, p)) == bases(0.7, θ, nothing)
    @inferred bases(0.7, rand(), EQL.Parameters(θ, p))

    bases = PolynomialBasis(1, 100)
    p = nothing
    θ = rand(length(bases.bases))
    @test bases(0.7, θ, p) ≈ sum(θ[i] * 0.7^i for i in 1:length(bases.bases))
    @inferred bases(0.7, θ, p)
    @test bases(0.7, rand(), rand(), EQL.Parameters(θ, p)) == bases(0.7, θ, p)
    @inferred bases(0.7, rand(), rand(), EQL.Parameters(θ, p))
    @test bases(0.7, rand(), EQL.Parameters(θ, p)) == bases(0.7, θ, nothing)
    @inferred bases(0.7, rand(), EQL.Parameters(θ, p))

    bases = BasisSet(
        (u, p) -> 0.7p[1] * u,
        (u, p) -> p[1] * u^2 / 3,
        (u, p) -> (p[1] + p[2]) * u^3
    )
    p = (0.3, -0.5)
    θ = [1.7, 0.5, -1.3]
    @test bases(1.7, θ, p) ≈ 0.7 * 0.3 * 1.7 * 1.7 + 0.3 * 1.7^2 / 3 * 0.5 - 1.3 * (0.3 - 0.5) * 1.7^3
end

@testset "eval_and_diff" begin
    bases = BasisSet(
        (u, p) -> 0.7p[1] * u,
        (u, p) -> p[1] * u^2 / 3,
        (u, p) -> (p[1] + p[2]) * u^3
    )
    p = (0.3, -0.5)
    gu, ∂gu = EQL.eval_and_diff(bases, 0.35, p, 1)
    @test gu ≈ 0.7 * 0.3 * 0.35
    @test ∂gu ≈ 0.7 * 0.3
    gu, ∂gu = EQL.eval_and_diff(bases, 0.7512, p, 2)
    @test gu ≈ 0.3 * 0.7512^2 / 3
    @test ∂gu ≈ 0.3 * 0.7512 * 2 / 3
    gu, ∂gu = EQL.eval_and_diff(bases, 0.7512, p, 3)
    @test gu ≈ (0.3 - 0.5) * 0.7512^3
    @test ∂gu ≈ (0.3 - 0.5) * 3 * 0.7512^2
    @inferred EQL.eval_and_diff(bases, 0.35, p, 1)
    @inferred EQL.eval_and_diff(bases, 0.7512, p, 2)
    @inferred EQL.eval_and_diff(bases, 0.7512, p, 3)
end