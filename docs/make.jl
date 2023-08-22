using StepwiseEQL
using Documenter
using Test
DocMeta.setdocmeta!(StepwiseEQL, :DocTestSetup, :(using StepwiseEQL, Test);
    recursive=true)
const IS_CI = get(ENV, "CI", "false") == "true"

const _PAGES = [
    "Home" => "index.md",
    "Installation" => "installation.md",
    "Repository Structure" => "repository_structure.md",
    "The Algorithm" => "algorithm.md",
    "Paper Results" => [
    "Overview" => "overview.md",
    "Reproducing Figure 2" => "case_studies/figure_2.md",
    "Reproducing Figure 3" => "case_studies/figure_3.md",
    "Case Study 1" => "case_studies/cs1.md",
    "Case Study 2" => "case_studies/cs2.md",
    "Case Study 3" => [
        "Accurate continuum limit" => "case_studies/cs3a.md",
        "Inaccurate continuum limit" => "case_studies/cs3b.md",
    ],
    "Case Study 4" => [
        "Accurate continuum limit" => "case_studies/cs4a.md",
        "Inaccurate continuum limit" => "case_studies/cs4b.md",
    ],
    "Discrete Densities at the Boundaries" => "supplementary_material/new_density.md",
    "A Piecewise Proliferation Law" => "supplementary_material/piecewise_prof.md",
    "Linear Diffusion" => "supplementary_material/linear_diffusion.md",
    "Parameter Sensitivity Study" => "supplementary_material/parameter_sensitivity.md"
    ],
]

# Make sure we haven't forgotten any files
set = Set{String}()
for page in _PAGES
    if page[2] isa String
        push!(set, normpath(page[2]))
    else
        for _page in page[2]
            if _page[2] isa String
                push!(set, normpath(_page[2]))
            else
                for __page in _page[2]
                    push!(set, normpath(__page[2]))
                end
            end
        end
    end
end
missing_set = String[]
doc_dir = joinpath(@__DIR__, "src", "")
for (root, dir, files) in walkdir(doc_dir)
    for file in files
        filename = normpath(replace(joinpath(root, file), doc_dir => ""))
        if endswith(filename, ".md") && filename ∉ set
            push!(missing_set, filename)
        end
    end
end
!isempty(missing_set) && error("Missing files: $missing_set")

# Make and deploy
makedocs(;
    modules=[StepwiseEQL],
    authors="Daniel VandenHeuvel <danj.vandenheuvel@gmail.com>",
    repo="https://github.com/DanielVandH/StepwiseEQL.jl/blob/{commit}{path}#{line}",
    sitename="StepwiseEQL.jl",
    format=Documenter.HTML(;
        prettyurls=IS_CI,
        canonical="https://DanielVandH.github.io/StepwiseEQL.jl",
        edit_link="main",
        collapselevel=1,
        assets=String[]),
    pages=_PAGES
)
deploydocs(;
    repo="github.com/DanielVandH/StepwiseEQL.jl",
    devbranch="main",
    versions=nothing)