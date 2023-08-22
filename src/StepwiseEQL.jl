module StepwiseEQL

using EpithelialDynamics1D
using FiniteVolumeMethod1D
using MovingBoundaryProblems1D
using Unrolled
using ForwardDiff
using DiffResults
using SciMLBase
using LinearSolve
using ElasticArrays
using DataInterpolations
using LinearAlgebra
using StatsBase
using Random
using FLoops
using OrdinaryDiffEq
using PrettyTables
using BlockDiagonals
using Printf
using Setfield
using Bijections
using LaTeXStrings
using Memoize

export BasisSet, PolynomialBasis, stepwise_selection, AveragedODESolution, default_loss, latex_table

include("structs/eql_model.jl")
include("structs/eql_solution.jl")
include("structs/ensemble_eql_solution.jl")
include("structs/averaged_ode_solution.jl")

include("function_evaluation/basis_functions.jl")
include("function_evaluation/density_computations.jl")

include("problem_building/matrix_construction.jl")
include("problem_building/pde_construction.jl")

include("individual_steps/cross_validation.jl")
include("individual_steps/density_loss.jl")
include("individual_steps/evaluate_loss.jl")
include("individual_steps/model_voting.jl")
include("individual_steps/penalty.jl")
include("individual_steps/regression_loss.jl")

include("algorithm/stepwise.jl")
include("algorithm/run_stepwise.jl")
include("algorithm/model_sampling.jl")

function subscriptnumber(i::Int)
    if i < 0
        c = [Char(0x208B)]
    else
        c = []
    end
    for d in reverse(digits(abs(i)))
        push!(c, Char(0x2080 + d))
    end
    return join(c)
end

const FLOOPS_EX = SequentialEx() # serial: do ThreadedEx() for parallel

end