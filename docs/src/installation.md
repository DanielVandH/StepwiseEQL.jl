# Installation 
To install the package, you first need to install Julia. To do this:

1. You can download Julia [here](https://julialang.org/downloads/), preferably `v1.9.0` (the latest version at the time of writing). My preferred installer for Julia is [juliaup](https://github.com/JuliaLang/juliaup).
2. You need an actual editor for Julia, e.g. [VS Code](https://code.visualstudio.com/) with the [Julia extension](https://code.visualstudio.com/docs/languages/julia).

To now actually install the package, there are two ways.

## Installing from GitHub

The first way would be to install it as you would any other package. Within Julia, you can do:
```julia
using Pkg 
Pkg.add("https://github.com/DanielVandH/StepwiseEQL.jl")
using StepwiseEQL 
```
With this, you will have access to the functions required for running the examples in the documentation and thus reproduce the results in the paper; you will need to install the packages listed 
in those examples of course. For example, typing 
```julia
using Pkg
Pkg.add(["CairoMakie", "LinearSolve"])
using CairoMakie, LinearSolve
```

will install and load CairoMakie.jl and LinearSolve.jl.

## Cloning from GitHub

Alternatively, you could clone the repository so that you have the entire package downloaded. With this done, you could for example open the package's folder in VS Code. You can then do:
```julia
using Pkg 
cd(@__DIR__)
Pkg.activate(".") 
Pkg.resolve()
Pkg.instantiate()
using StepwiseEQL 
```

This will activate the package in your REPL, and will also download all the dependencies (packages) needed for the examples.
