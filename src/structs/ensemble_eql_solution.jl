struct EnsembleEQLSolution{M}
    solutions::Vector{M}
    final_loss::Dict{Vector{Bool},Tuple{Float64,Int}}
    best_model::Vector{Bool}
    best_model_indices::Vector{Int} # indices of the best model in the solutions vector
end
function Base.show(io::IO, mime::MIME"text/plain", eql_sol::EnsembleEQLSolution)
    println(io, "EnsembleEQL Solution with $(length(eql_sol.solutions)) solutions")
    println(io, "Best model: $(eql_sol.best_model)")
    show(io, mime, eql_sol.final_loss)
end
function EnsembleEQLSolution(eql_sol::Vector{M}) where {M}
    final_loss = Dict{Vector{Bool},Tuple{Float64,Int}}()
    for _eql_sol in eql_sol
        final_indicator = _eql_sol.indicator_history[:, end]
        if !haskey(final_loss, final_indicator)
            final_loss[final_indicator] = (_eql_sol.loss_history[end], 1)
        else
            final_loss[final_indicator] = (final_loss[final_indicator][1] + _eql_sol.loss_history[end], final_loss[final_indicator][2] + 1)
        end
    end
    averaged_loss = Dict{Vector{Bool},Float64}()
    for (model, (total_loss, num_occurrences)) in final_loss
        averaged_loss[model] = total_loss / num_occurrences
        final_loss[model] = (total_loss / num_occurrences, num_occurrences)
    end
    _, best_model = findmin(averaged_loss)
    best_model_indices = findall(eql_sol) do sol
        sol.indicator_history[:, end] == best_model
    end
    return EnsembleEQLSolution(eql_sol, final_loss, best_model, best_model_indices)
end