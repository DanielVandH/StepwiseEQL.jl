using StepwiseEQL
using Test
using SafeTestsets

@testset verbose = true "StepwiseEQL" begin
    @safetestset "Basis" begin
        include("bases.jl")
    end
    @safetestset "Density" begin
        include("density.jl")
    end
    @safetestset "Basis System" begin
        include("basis_system.jl")
    end
    @safetestset "PDE Construction" begin
        include("pde_construction.jl")
    end
    @safetestset "Loss Function" begin
        include("loss_function.jl")
    end
    @safetestset "Individual Steps" begin
        include("individual_steps.jl")
    end
    @safetestset "Detailed Stepwise EQL for Diffusion" begin
        include("stepwise_diffusion.jl")
    end
    @safetestset "Detailed Stepwise EQL for Proliferation" begin
        include("stepwise_proliferation.jl")
    end
    @safetestset "Detailed Stepwise EQL for Averaged Proliferation" begin
        include("stepwise_averaged.jl")
    end
end
