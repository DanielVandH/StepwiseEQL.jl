# Repository Structure 

```@contents
Pages = ["repository_structure.md"]
```

In this section, we describe the structure of the repository. Note that the description of the functions used for running the algorithm itself are defined in the next section.

## Root 
The root of the repository contains five files.

- `.gitignore`: This file tells git which files to ignore when committing changes.
- `LICENSE`: This file contains the license for the repository. The license for this package is an MIT license. 
- `README.md`: This file contains the basic description of the repository, primarily for linking to the documentation.
- `Project.toml`: This file contains the dependencies for the package.
- `Manifest.toml`: This file contains the exact versions of the dependencies for the package. This together with Project.toml makes the paper results exactly reproducible.

## `.github/workflows`
This folder just contains some basic workflows for testing the package and producing its documentation. These are standard files you'll see in almost any Julia package.

## `docs`
The `docs` folder is what produces the documentation, and produces the paper results. Within this folder, what actually produces the documentation is the `make.jl` file, which uses [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) and [Literate.jl](https://github.com/fredrikekre/Literate.jl) to run the scripts and launch the documentation. The `src` folders contains all the scripts for each section, which you can inspect yourself - just read the documentation and, if you do want to see the source file, either see the link at the top-right or at the bottom of the page.

## `src`
The `src` folder is where all the code for the algorithm lives. The folder is broken into other folders, and the only file inside this folder directly is:

- `StepwiseEQL.jl`: This is the module that defines the package and loads in all the dependencies and the files.

### `src/structs`
The `src/structs` folder contains the definitions for all the structs used in the package:

- `eql_model.jl`: The definition of an `EQLModel`, used for producing the stepwise learning results. This struct would have a very similar definition if you were to extend to other types of problems.
- `eql_solution.jl`: The definition of an `EQLSolution`, giving the results for the stepwise learning procedure.
- `ensemble_eql_solution.jl`: If running the stepwise learning procedure for multiple initial sets of active coefficients, you obtain an `EnsembleEQLSolution` defined here.
- `averaged_ode_solution.jl`: For proliferation we average the solution over each simulation, leading to a single set of results. This average is defined as an `AveragedODESolution`, defined here.

### `src/function_evaluation`
The `src/function_evaluation` folder contains the files used for defining how we evaluate basis functions, and for how we compute densities and their derivatives:

- `basis_functions.jl`: Defines the structs for a basis function, specifically as a `BasisSet` or a `PolynomialBasis`.
- `density_computations.jl`: This contains all the functions that are used to compute densities, their derivatives, and similarly for leading edges.

### `src/problem_building`
The `src/problem_building` folder contains the main functions used for building up the `EQLModel`s:

- `matrix_construction.jl`: The main component of the algorithm is the construction of the matrix system $\boldsymbol A\boldsymbol\theta = \boldsymbol b$. The code in this file constructs the matrix $\boldsymbol A$ and the vector $\boldsymbol b$.
- `pde_construction.jl`: This script automatically detects the type of PDE to be built (fixed boundary, moving boundary, proliferation, etc.) and constructs it.

### `src/individual_steps`
The `src/individual_steps` folder is used for performing the individual steps in our procedure:

- `cross_validation.jl`: This script contains the code used for creating the training and test sets if cross-validation is used for the algorithm.
- `density_loss.jl`: This script contains the functions used for evaluating the density component, and the leading edge component, of the loss function.
- `evaluate_loss.jl`: This script contains the functions used for evaluating the loss function completely.
- `model_voting.jl`: This script contains the functions used for incorporating and removing each term one at a time during an individual step of the algorithm, and for then making a vote on which model to step to in the next iteration.
- `penalty.jl`: This file contains the functions used for penalising model complexity, and also for enforcing constraints on $D(q)$ and $E(q)$.
- `regression_loss.jl`: This file contains the functions used for evaluating the regression component of the loss function.

### `src/algorithm`
The `src/algorithm` folder is used for actually running the algorithm, with the main script being `stepwise.jl`:

- `stepwise.jl`: This script contains the main entry point for running our stepwise procedure, defining the `stepwise_selection` function (described in more detail in the next section). 
- `run_stepwise.jl`: The `stepwise_selection` function from the previous point constructs an `EQLModel` and then runs the main internal method of `stepwise_selection` that actually runs all the results. This is defined in this file.
- `model_sampling.jl`: This file contains the internal function that is used for running the procedure for multiple initial conditions efficiently.

## `test` 
This folder is just for testing the algorithm's implementation itself. It is a bit lengthy and unlikely to be very readable, so you should not be overly concerned about it. If you did want to test the function, you can run 

```julia
julia> ] test
```

assuming the `StepwiseEQL` package is activated; you can tell if it is by seeing if, after typing `] test` inside the Julia REPL without executing it, you see

```julia
(StepwiseEQL) pkg> test
```