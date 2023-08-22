function model_history(model, indicators)
    θ = zeros(size(indicators))
    for (i, model_subset) in (enumerate ∘ eachcol)(indicators)
        θ_subset = projected_solve(model.A, model.b, model_subset)
        θ[:, i] .= θ_subset
    end
    return θ
end

function loss_history(model, indicators, loss_function; use_relative_err=true, leading_edge_error=true, extrapolate_pde=false, conserve_mass=false)
    regression_loss = zeros(size(indicators, 2))
    density_loss = zeros(size(indicators, 2))
    loss = zeros(size(indicators, 2))
    for (i, model_subset) in (enumerate ∘ eachcol)(indicators)
        _regression_loss, _density_loss = evaluate_loss(model, model_subset; cross_validation=false, use_relative_err, leading_edge_error, extrapolate_pde, conserve_mass)
        loss[i] = loss_function(_regression_loss, _density_loss, model_subset)
        regression_loss[i] = _regression_loss
        density_loss[i] = _density_loss
    end
    return regression_loss, density_loss, loss
end

Base.@kwdef struct EQLSolution{M,P,S}
    model::M
    diffusion_theta::Vector{Float64}
    reaction_theta::Vector{Float64}
    rhs_theta::Vector{Float64}
    moving_boundary_theta::Vector{Float64}
    diffusion_subset::Vector{Int}
    reaction_subset::Vector{Int}
    rhs_subset::Vector{Int}
    moving_boundary_subset::Vector{Int}
    diffusion_theta_history::Matrix{Float64}
    reaction_theta_history::Matrix{Float64}
    rhs_theta_history::Matrix{Float64}
    moving_boundary_theta_history::Matrix{Float64}
    diffusion_subset_history::Matrix{Bool}
    reaction_subset_history::Matrix{Bool}
    rhs_subset_history::Matrix{Bool}
    moving_boundary_subset_history::Matrix{Bool}
    diffusion_vote_history::Matrix{Float64}
    reaction_vote_history::Matrix{Float64}
    rhs_vote_history::Matrix{Float64}
    moving_boundary_vote_history::Matrix{Float64}
    regression_loss_history::Vector{Float64}
    density_loss_history::Vector{Float64}
    loss_history::Vector{Float64}
    indicator_history::Matrix{Bool}
    vote_history::Matrix{Float64}
    pde::P
    pde_sol::S
end
function EQLSolution(model, indicator_history, vote_history, loss_function; use_relative_err=true, leading_edge_error=true, extrapolate_pde=false, conserve_mass=false)
    indicator_history = Matrix(indicator_history)
    vote_history = Matrix(vote_history)
    θ_history = model_history(model, indicator_history)
    nd = num_diffusion(model)
    nr = num_reaction(model)
    nh = num_rhs(model)
    ne = !conserve_mass ? num_moving_boundary(model) : 0
    diffusion_subset = indicator_history[1:nd, end] |> findall
    reaction_subset = indicator_history[(nd+1):(nd+nr), end] |> findall
    rhs_subset = indicator_history[(nd+nr+1):(nd+nr+nh), end] |> findall
    moving_boundary_subset = indicator_history[(nd+nr+nh+1):(nd+nr+nh+ne), end] |> findall
    diffusion_theta = θ_history[1:nd, end]
    reaction_theta = θ_history[(nd+1):(nd+nr), end]
    rhs_theta = θ_history[(nd+nr+1):(nd+nr+nh), end]
    moving_boundary_theta = θ_history[(nd+nr+nh+1):(nd+nr+nh+ne), end]
    diffusion_theta_history = θ_history[1:nd, :]
    reaction_theta_history = θ_history[(nd+1):(nd+nr), :]
    rhs_theta_history = θ_history[(nd+nr+1):(nd+nr+nh), :]
    moving_boundary_theta_history = θ_history[(nd+nr+nh+1):(nd+nr+nh+ne), :]
    diffusion_subset_history = indicator_history[1:nd, :]
    reaction_subset_history = indicator_history[(nd+1):(nd+nr), :]
    rhs_subset_history = indicator_history[(nd+nr+1):(nd+nr+nh), :]
    moving_boundary_subset_history = indicator_history[(nd+nr+nh+1):(nd+nr+nh+ne), :]
    diffusion_vote_history = vote_history[1:nd, :]
    reaction_vote_history = vote_history[(nd+1):(nd+nr), :]
    rhs_vote_history = vote_history[(nd+nr+1):(nd+nr+nh), :]
    moving_boundary_vote_history = vote_history[(nd+nr+nh+1):(nd+nr+nh+ne), :]
    regression_loss_history, density_loss_history, _loss_history = loss_history(model, indicator_history, loss_function; use_relative_err, leading_edge_error, extrapolate_pde, conserve_mass)
    pde = rebuild_pde(model, θ_history[:, end], indicator_history[:, end], conserve_mass)
    pde_sol = solve(pde, TRBDF2(linsolve=KLUFactorization()), saveat=get_saveat(model.cell_sol), verbose=false)
    return EQLSolution(;
        model,
        diffusion_theta,
        reaction_theta,
        rhs_theta,
        moving_boundary_theta,
        diffusion_subset,
        reaction_subset,
        rhs_subset,
        moving_boundary_subset,
        diffusion_theta_history,
        reaction_theta_history,
        rhs_theta_history,
        moving_boundary_theta_history,
        diffusion_subset_history,
        reaction_subset_history,
        rhs_subset_history,
        moving_boundary_subset_history,
        diffusion_vote_history,
        reaction_vote_history,
        rhs_vote_history,
        moving_boundary_vote_history,
        regression_loss_history,
        density_loss_history,
        loss_history=_loss_history,
        indicator_history,
        vote_history,
        pde,
        pde_sol
    )
end
num_steps(eql_sol::EQLSolution) = length(eql_sol.loss_history)

function Base.show(io::IO, ::MIME"text/plain", eql_sol::EQLSolution;
    step_limit=6,
    crop=:horizontal,
    backend=Val(:text),
    booktabs=false,
    show_votes=false,
    show_all_loss=false,
    crayon=Crayon(bold=true, foreground=:green),
    latex_crayon=["color{blue}", "textbf"],
    transpose=true)
    println(io, "StepwiseEQL Solution.")
    model = eql_sol.model 
    num_diffusion(model) > 0 && println(io, format_equation(eql_sol, :diffusion))
    num_reaction(model) > 0 && println(io, format_equation(eql_sol, :reaction))
    is_mb = model.pde_template isa MBProblem
    is_mb && num_rhs(model) > 0 && println(io, format_equation(eql_sol, :rhs))
    is_not_conserve = !isempty(eql_sol.moving_boundary_theta_history) # also check for conserve_mass
    is_mb && is_not_conserve && println(io, format_equation(eql_sol, :moving_boundary))
    stepwise_res, header_names = get_solution_table(eql_sol; backend, step_limit, show_votes, show_all_loss)
    body_hlines = get_body_hlines(eql_sol, stepwise_res, transpose)
    vlines = get_vlines(eql_sol, stepwise_res, transpose)
    if transpose
        old_header_names = header_names
        header_names = vcat("Step", stepwise_res[:, 1])
        stepwise_res = hcat(old_header_names[2:end], permutedims(stepwise_res[:, 2:end], (2, 1))) # Puts the steps numbers into the (transposed) first column, and strips the θ coefficient names from the first row after transposing
        crop = crop == :horizontal ? :vertical : :horizontal
    end
    highlighters = get_highlighters(eql_sol, stepwise_res, header_names; backend, crayon, latex_crayon, show_all_loss, transpose)
    if backend == Val(:text)
        return pretty_table(io, stepwise_res;
            backend=backend,
            header=header_names,
            body_hlines=body_hlines,
            vlines=vlines,
            highlighters=highlighters,
            crop=crop
        )
    else
        return pretty_table(io, stepwise_res;
            backend=backend,
            header=header_names,
            body_hlines=body_hlines,
            vlines=vlines,
            highlighters=highlighters,
            tf=booktabs ? tf_latex_booktabs : tf_latex_default
        )
    end
end
function format_equation(eql_sol::EQLSolution, mechanism)
    subset, sym, sup = if mechanism == :diffusion 
        eql_sol.diffusion_subset, "D", "ᵈ"
    elseif mechanism == :reaction
        eql_sol.reaction_subset, "R", "ʳ"
    elseif mechanism == :rhs
        eql_sol.rhs_subset, "H", "ʰ"
    elseif mechanism == :moving_boundary
        eql_sol.moving_boundary_subset, "E", "ᵉ"
    else
        throw(ArgumentError("Invalid mechanism: $mechanism"))
    end
    eqn = "    $sym(q) = "
    for i in subset
        eqn *= "θ" * subscriptnumber(i) * "$sup ϕ" * subscriptnumber(i) * "$sup(q)"
        if i ≠ subset[end]
            eqn *= " + "
        end
    end
    if length(subset) == 0
        eqn *= "0"
    end
    return eqn
end

function str_format(num, backend, threshold=0.001) # the Boolean is to detect if a LatexCell is needed
    if 0 < abs(num) < threshold
        if backend == Val(:text)
            return @sprintf("%.2e", num), false
        elseif backend == Val(:latex)
            str = @sprintf("%.2e", num)
            return raw"\num{" * str * "}", true
        end
    else
        @sprintf("%.2f", num), false
    end
end
function get_solution_table(eql_sol::EQLSolution; backend=Val(:text), step_limit=6, show_votes=true, show_all_loss=true)
    if backend == Val(:text)
        diff_θ_names = ["θ" * subscriptnumber(i) * (show_votes ? "ᵈ (votes)" : "ᵈ") for i in 1:num_diffusion(eql_sol.model)]
        react_θ_names = ["θ" * subscriptnumber(i) * (show_votes ? "ʳ (votes)" : "ʳ") for i in 1:num_reaction(eql_sol.model)]
        if eql_sol.model.pde_template isa MBProblem
            rhs_θ_names = ["θ" * subscriptnumber(i) * (show_votes ? "ʰ (votes)" : "ʰ") for i in 1:num_rhs(eql_sol.model)]
            if !isempty(eql_sol.moving_boundary_theta_history)
                moving_boundary_θ_names = ["θ" * subscriptnumber(i) * (show_votes ? "ᵉ (votes)" : "ᵉ") for i in 1:num_moving_boundary(eql_sol.model)]
            else
                moving_boundary_θ_names = String[]
            end
        end
    else
        diff_θ_names = LatexCell.([L"$\theta_{%$(i)}^d$ " * (show_votes ? "(votes)" : "") for i in 1:num_diffusion(eql_sol.model)])
        react_θ_names = LatexCell.([L"$\theta_{%$(i)}^r$ " * (show_votes ? "(votes)" : "") for i in 1:num_reaction(eql_sol.model)])
        if eql_sol.model.pde_template isa MBProblem
            rhs_θ_names = LatexCell.([L"$\theta_{%$(i)}^h$ " * (show_votes ? "(votes)" : "") for i in 1:num_rhs(eql_sol.model)])
            if !isempty(eql_sol.moving_boundary_theta_history) # check for conserve_mass
                moving_boundary_θ_names = LatexCell.([L"$\theta_{%$(i)}^e$ " * (show_votes ? "(votes)" : "") for i in 1:num_moving_boundary(eql_sol.model)])
            else
                moving_boundary_θ_names = LatexCell[]
            end
        end
    end
    if !(eql_sol.model.pde_template isa MBProblem)
        if show_all_loss
            row_names = vcat(diff_θ_names, react_θ_names, "Regression Loss", "Density Loss", "Loss")
        else
            row_names = vcat(diff_θ_names, react_θ_names, "Loss")
        end
    else
        if show_all_loss
            row_names = vcat(diff_θ_names, react_θ_names, rhs_θ_names, moving_boundary_θ_names, "Regression Loss", "Density Loss", "Loss")
        else
            row_names = vcat(diff_θ_names, react_θ_names, rhs_θ_names, moving_boundary_θ_names, "Loss")
        end
    end
    header_names = vcat("Coefficient", ["$i" for i in 1:num_steps(eql_sol)])
    stepwise_res = Matrix{Any}(undef, length(row_names), length(header_names))
    stepwise_res[:, 1] .= row_names
    # Yes, the below could be easily simplified by creating a general function that accepts 
    # stepwise_res, the history, the names, etc. and then just calling it for each of the
    # mechanisms.
    for j in 2:(1+num_steps(eql_sol)) # intended due to the row names
        for i in 1:length(diff_θ_names)
            θ = eql_sol.diffusion_theta_history[i, j-1]
            vote = eql_sol.diffusion_vote_history[i, j-1]
            formatted_θ, latex_needed = str_format(θ, backend)
            formatted_vote = @sprintf("%.2f", vote)
            if !latex_needed
                stepwise_res[i, j] = formatted_θ * (show_votes ? (" (" * formatted_vote * ")") : "")
            else
                stepwise_res[i, j] = LatexCell(formatted_θ * (show_votes ? (" (" * formatted_vote * ")") : ""))
            end
        end
        for i in (length(diff_θ_names)+1):(length(diff_θ_names)+length(react_θ_names))
            θ = eql_sol.reaction_theta_history[i-length(diff_θ_names), j-1]
            vote = eql_sol.reaction_vote_history[i-length(diff_θ_names), j-1]
            formatted_θ, latex_needed = str_format(θ, backend)
            formatted_vote = @sprintf("%.2f", vote)
            if !latex_needed
                stepwise_res[i, j] = formatted_θ * (show_votes ? (" (" * formatted_vote * ")") : "")
            else
                stepwise_res[i, j] = LatexCell(formatted_θ * (show_votes ? (" (" * formatted_vote * ")") : ""))
            end
        end
        if eql_sol.model.pde_template isa MBProblem
            for i in (length(diff_θ_names)+length(react_θ_names)+1):(length(diff_θ_names)+length(react_θ_names)+length(rhs_θ_names))
                θ = eql_sol.rhs_theta_history[i-length(diff_θ_names)-length(react_θ_names), j-1]
                vote = eql_sol.rhs_vote_history[i-length(diff_θ_names)-length(react_θ_names), j-1]
                formatted_θ, latex_needed = str_format(θ, backend)
                formatted_vote = @sprintf("%.2f", vote)
                if !latex_needed
                    stepwise_res[i, j] = formatted_θ * (show_votes ? (" (" * formatted_vote * ")") : "")
                else
                    stepwise_res[i, j] = LatexCell(formatted_θ * (show_votes ? (" (" * formatted_vote * ")") : ""))
                end
            end
            try # in case of conserve mass
                for i in (length(diff_θ_names)+length(react_θ_names)+length(rhs_θ_names)+1):(length(diff_θ_names)+length(react_θ_names)+length(rhs_θ_names)+length(moving_boundary_θ_names))
                    θ = eql_sol.moving_boundary_theta_history[i-length(diff_θ_names)-length(react_θ_names)-length(rhs_θ_names), j-1]
                    vote = eql_sol.moving_boundary_vote_history[i-length(diff_θ_names)-length(react_θ_names)-length(rhs_θ_names), j-1]
                    formatted_θ, latex_needed = str_format(θ, backend)
                    formatted_vote = @sprintf("%.2f", vote)
                    if !latex_needed
                        stepwise_res[i, j] = formatted_θ * (show_votes ? (" (" * formatted_vote * ")") : "")
                    else
                        stepwise_res[i, j] = LatexCell(formatted_θ * (show_votes ? (" (" * formatted_vote * ")") : ""))
                    end
                end
            catch
            end
        end
        if show_all_loss
            regression_loss = eql_sol.regression_loss_history[j-1]
            formatted_regression_loss, latex_needed = str_format(regression_loss, backend)
            if !latex_needed
                stepwise_res[end-2, j] = formatted_regression_loss
            else
                stepwise_res[end-2, j] = LatexCell(formatted_regression_loss)
            end
            density_loss = eql_sol.density_loss_history[j-1]
            formatted_density_loss, latex_needed = str_format(density_loss, backend)
            if !latex_needed
                stepwise_res[end-1, j] = formatted_density_loss
            else
                stepwise_res[end-1, j] = LatexCell(formatted_density_loss)
            end
            loss = eql_sol.loss_history[j-1]
            formatted_loss, latex_needed = str_format(loss, backend)
            if !latex_needed
                stepwise_res[end, j] = formatted_loss
            else
                stepwise_res[end, j] = LatexCell(formatted_loss)
            end
        else
            loss = eql_sol.loss_history[j-1]
            if isfinite(loss) || backend == Val(:text)
                formatted_loss, latex_needed = str_format(loss, backend)
                if !latex_needed
                    stepwise_res[end, j] = formatted_loss
                else
                    stepwise_res[end, j] = LatexCell(formatted_loss)
                end
            else
                stepwise_res[end, j] = L"\infty"
            end
        end
    end
    if size(stepwise_res, 2) > step_limit + 1
        if backend == Val(:text)
            stepwise_res = hcat(stepwise_res[:, 1:(step_limit÷2)], fill("⋯", size(stepwise_res, 1), 1), stepwise_res[:, end-((step_limit÷2)-1):end])
        else
            stepwise_res = hcat(stepwise_res[:, 1:(step_limit÷2)], LatexCell.(fill(L"$\cdots$", size(stepwise_res, 1), 1)), stepwise_res[:, end-((step_limit÷2)-1):end])
        end
        header_names = vcat(header_names[1:(step_limit÷2)], "⋯", header_names[end-((step_limit÷2)-1):end])
    end
    return stepwise_res, header_names
end

function get_highlighter(data, i, j; eql_sol, table, header_names, fnc=:diffusion, show_all_loss=true, transpose=false)
    votes = eql_sol.vote_history
    if !transpose
        intable = j > 1 && i < size(table, 1) - (show_all_loss ? 2 : 0) # j = 1 is the Coefficient column, and ≥ size(table, 1) - 2 is the Regression loss, Density loss, and Loss rows
    else
        intable = j > 1 && j < size(table, 2) - (show_all_loss ? 2 : 0) # j = 1 is the Coefficient column, and ≥ size(table, 1) - 2 is the Regression loss, Density loss, and Loss rows
    end
    if !transpose
        notlast = j < size(table, 2) # don't need to highlight the final step 
    else
        notlast = i < size(table, 1)
    end
    inmechanism = if fnc == :diffusion
        is_diffusion_index(eql_sol.model, transpose ? j - 1 : i)
    elseif fnc == :reaction
        is_reaction_index(eql_sol.model, transpose ? j - 1 : i)
    elseif fnc == :rhs
        is_rhs_index(eql_sol.model, transpose ? j - 1 : i)
    elseif fnc == :moving_boundary # need to also check for conserve_mass
        !isempty(eql_sol.moving_boundary_theta_history) && is_moving_boundary_index(eql_sol.model, transpose ? j - 1 : i)
    else
        throw(ArgumentError("Invalid function specified for highlighter: $fnc"))
    end
    possible_highlight = intable && notlast && inmechanism
    if possible_highlight
        if !transpose
            header_names[j] == "⋯" && return false
            step = parse(Int, header_names[j])
            max_vote = maximum(abs.(votes[:, step]))
            highlight = abs(votes[i, step]) == max_vote
        else
            table[i, 1] == "⋯" && return false
            step = parse(Int, table[i, 1])
            max_vote = maximum(abs.(votes[:, step]))
            highlight = abs(votes[j-1, step]) == max_vote
        end
    end
    return possible_highlight && highlight
end
function get_highlighters(eql_sol::EQLSolution, stepwise_res, header_names;
    backend=Val(:text),
    crayon=Crayon(bold=true, foreground=:green),
    latex_crayon=["color{blue}", "bfseries"],
    show_all_loss=true,
    transpose=false)
    if backend == Val(:text)
        h1 = Highlighter(
            f=(data, i, j) -> get_highlighter(data, i, j; eql_sol=eql_sol, table=stepwise_res, header_names=header_names, fnc=:diffusion, show_all_loss, transpose),
            crayon=crayon)
        h2 = Highlighter(
            f=(data, i, j) -> get_highlighter(data, i, j; eql_sol=eql_sol, table=stepwise_res, header_names=header_names, fnc=:reaction, show_all_loss, transpose),
            crayon=crayon)
        if eql_sol.model.pde_template isa MBProblem
            h3 = Highlighter(
                f=(data, i, j) -> get_highlighter(data, i, j; eql_sol=eql_sol, table=stepwise_res, header_names=header_names, fnc=:rhs, show_all_loss, transpose),
                crayon=crayon)
            h4 = Highlighter(
                f=(data, i, j) -> get_highlighter(data, i, j; eql_sol=eql_sol, table=stepwise_res, header_names=header_names, fnc=:moving_boundary, show_all_loss, transpose),
                crayon=crayon)
        end
        if eql_sol.model.pde_template isa MBProblem
            return (h1, h2, h3, h4)
        else
            return (h1, h2)
        end
    elseif backend == Val(:latex)
        h1 = LatexHighlighter(
            (data, i, j) -> get_highlighter(data, i, j; eql_sol=eql_sol, table=stepwise_res, header_names=header_names, fnc=:diffusion, show_all_loss, transpose),
            latex_crayon)
        h2 = LatexHighlighter(
            (data, i, j) -> get_highlighter(data, i, j; eql_sol=eql_sol, table=stepwise_res, header_names=header_names, fnc=:reaction, show_all_loss, transpose),
            latex_crayon)
        if eql_sol.model.pde_template isa MBProblem
            h3 = LatexHighlighter(
                (data, i, j) -> get_highlighter(data, i, j; eql_sol=eql_sol, table=stepwise_res, header_names=header_names, fnc=:rhs, show_all_loss, transpose),
                latex_crayon)
            h4 = LatexHighlighter(
                (data, i, j) -> get_highlighter(data, i, j; eql_sol=eql_sol, table=stepwise_res, header_names=header_names, fnc=:moving_boundary, show_all_loss, transpose),
                latex_crayon)
        end
        if eql_sol.model.pde_template isa MBProblem
            return (h1, h2, h3, h4)
        else
            return (h1, h2)
        end
    else
        throw(ArgumentError("Invalid backend: $backend"))
    end
end
function get_body_hlines(eql_sol::EQLSolution, stepwise_res, transpose=false)
    if !transpose
        unique!([
            num_diffusion(eql_sol.model),
            num_diffusion(eql_sol.model) + num_reaction(eql_sol.model),
            num_diffusion(eql_sol.model) + num_reaction(eql_sol.model) + num_rhs(eql_sol.model),
            num_diffusion(eql_sol.model) + num_reaction(eql_sol.model) + num_rhs(eql_sol.model) + num_moving_boundary(eql_sol.model)
        ]) # unique in case we have num_X = 0 for any X ∈ (diffusion, reaction, rhs, moving_boundary)
    else
        return [0]
    end
end
function get_vlines(eql_sol, stepwise_res, transpose=false)
    if !transpose
        return [0, 1, size(stepwise_res, 2)]
    else
        arr = 1 .+ unique!([
            num_diffusion(eql_sol.model),
            num_diffusion(eql_sol.model) + num_reaction(eql_sol.model),
            num_diffusion(eql_sol.model) + num_reaction(eql_sol.model) + num_rhs(eql_sol.model),
            num_diffusion(eql_sol.model) + num_reaction(eql_sol.model) + num_rhs(eql_sol.model) + num_moving_boundary(eql_sol.model)
        ])
        return vcat(0, 1, arr, size(stepwise_res, 1) + 1) |> unique
    end
end

"""
    latex_table(eql_sol::EQLSolution; kwargs...)

Prints the `EQLSolution` `eql_sol` in LaTeX format. Keyword arguments are passed to `show`, which 
has signature 

    Base.show(io::IO, ::MIME"text/plain", eql_sol::EQLSolution;
        step_limit=6,
        crop=:horizontal,
        backend=Val(:text),
        booktabs=true,
        show_votes=false,
        show_all_loss=false,
        crayon=Crayon(bold=true, foreground=:green),
        latex_crayon=["color{blue}", "textbf"],
        transpose=true)

(Of course, `backend=Val(:latex)` is passed - you cannot change that.)
"""
function latex_table(eql_sol::EQLSolution; kwargs...)
    show(stdout, MIME"text/plain"(), eql_sol; backend=Val(:latex), kwargs...)
end