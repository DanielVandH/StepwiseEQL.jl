```@meta
CurrentModule = StepwiseEQL
```

# The Algorithm 

```@contents
Pages = ["algorithm.md"]
```

In this section, we describe the functions used for actually running the algorithm. Here we only describe how to run the results after obtaining a `CellProblem`. If you want to know more about setting up the `CellProblem`s, or solving any PDEs, please see:

- [EpithelialDynamics1D.jl](https://github.com/DanielVandH/EpithelialDynamics1D.jl): This is the package that actually implements the discrete model of the epithelial dynamics. Please see its [documentation](https://danielvandh.github.io/EpithelialDynamics1D.jl/stable/) for detailed information and examples.
- [FiniteVolumeMethod1D.jl](https://github.com/DanielVandH/FiniteVolumeMethod1D.jl): This is the package that solves the PDEs on a fixed boundary. Please see its [documentation](https://danielvandh.github.io/FiniteVolumeMethod1D.jl/stable/) for detailed information and examples.
- [MovingBoundaryProblems1D.jl](https://github.com/DanielVandH/MovingBoundaryProblems1D.jl): This is the package that solves the PDEs on a moving boundary. Please see its [documentation](https://danielvandh.github.io/MovingBoundaryProblems1D.jl/stable/) for detailed information and examples.

## Basis Functions 
The functions below are used for defining basis functions:

```@docs
BasisSet 
PolynomialBasis
```

For example, the basis set $\{q, q^2, q^3\}$ can be defined in two ways:

```julia
using StepwiseEQL
f1 = (q, p) -> q 
f2 = (q, p) -> q^2 
f3 = (q, p) -> q^3 
B = BasisSet(f1, f2, f3)
```
```
(::BasisSet{Tuple{var"#307#308", var"#309#310", var"#311#312"}}) (generic function with 3 methods)
```

```julia
B = PolynomialBasis(1, 3)
```
```
(::BasisSet{Tuple{StepwiseEQL.var"#52#54"{Int64}, StepwiseEQL.var"#52#54"{Int64}, StepwiseEQL.var"#52#54"{Int64}}}) (generic function with 3 methods)
```

The reason for the second argument `p` is in case you want to give better control of the scaling on the basis function's coefficients. For example, the basis set $\{aq, bq^{-2}, cq^3\}$ can be defined as:

```julia
f1 = (q, p) -> p.a * q 
f2 = (q, p) -> p.b * q^(-2)
f3 = (q, p) -> p.c * q^3
B = BasisSet(f1, f2, f3)
```
```
(::BasisSet{Tuple{var"#313#314", var"#315#316", var"#317#318"}}) (generic function with 3 methods)
```

and you would then provide `p` when evaluating `B`, for example:

```julia 
p = (a = 2.0, b = 3.0, c = 5.0) # same p for each basis function
B(0.2, [0.3, 0.5, 1], p)
```
```
37.66
```

which returns $\theta_1aq + \theta_2bq^{-2} + \theta_3cq^3$ with $\theta_1 = 0.3$, $\theta_2 = 0.5$, $\theta_3 = 1$, $a = 2$, $b = 3$, $c = 5$, and $q = 0.2$.

## Averaging ODE Solutions 
Usually averaging over multiple realisations from the discrete model can be handled internally by the stepwise function, but you may want to re-average for the purpose of plotting (as we do in case studies 3 and 4). For this reason, we provide `AveragedODESolution` as an exported function that you can use:

```@docs
AveragedODESolution
```

## The Stepwise Algorithm
The entry point into our stepwise procedure is `stepwise_selection`, documented in detail below.

```@docs
stepwise_selection
```

## Printing LaTeX Tables 
From an `EQLSolution`, returned by `stepwise_selection`, you can print the LaTeX form of the table using `latex_table`:

```@docs
latex_table
```

## Loss Function
If you want to provide a new loss function, you can provide it as a keyword `loss_function` to `stepwise_selection`. The function that is used to construct the default loss is below for example.

```@docs
default_loss
```
