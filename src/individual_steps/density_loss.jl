@inline function solve_subproblem_pde(model::EQLModel{S,P}, θ_train, indicators=model.indicators, conserve_mass=false) where {S,P}
    pde = rebuild_pde(model, θ_train, indicators, conserve_mass)::P
    sol = solve(pde, TRBDF2(linsolve=KLUFactorization()), saveat=get_saveat(model.cell_sol), verbose=false)
    return sol
end

function evaluate_density_error(sol, model, j; use_relative_err=true)
    is_ensemble = model.cell_sol isa EnsembleSolution
    interp, L = if model.pde_template isa MBProblem
        @views LinearInterpolation{true}(sol.u[j][begin:(end-1)], model.pde_template.geometry.mesh_points), # ξᵢ ∈ [0, 1] for MBProblem
        sol.u[j][end]
    else
        LinearInterpolation{true}(sol.u[j], model.pde_template.geometry.mesh_points),
        1.0 # so that 1/L does not nothing
    end
    density_err = 0.0
    num_terms = 0
    if is_ensemble
        for k in model.simulation_indices
            cell_sol = model.cell_sol[k]
            for (i, r) in enumerate(cell_sol.u[j])
                L′ = r / L
                pde_q = max(0.0, interp(L′))
                if L′ > 1.0 && model.pde_template isa MBProblem
                    pde_q = 0.0
                end
                cell_q = cell_density(cell_sol, i, j)
                density_loss = (cell_q - pde_q)^2
                if use_relative_err
                    density_loss /= cell_q^2
                end
                density_err += density_loss
                num_terms += 1
            end
        end
    else
        cell_sol = model.cell_sol
        for (i, r) in enumerate(cell_sol.u[j])
            L′ = r / L
            pde_q = max(0.0, interp(L′))
            if L′ > 1.0 && model.pde_template isa MBProblem
                pde_q = 0.0
            end
            cell_q = cell_density(cell_sol, i, j)
            density_loss = (cell_q - pde_q)^2
            if use_relative_err
                density_loss /= cell_q^2
            end
            density_err += density_loss
            num_terms += 1
        end
    end
    return density_err, num_terms
end

function evaluate_leading_edge_error(sol, model, j; use_relative_err=true, leading_edge_error=true)
    if leading_edge_error
        is_ensemble = model.cell_sol isa EnsembleSolution
        leading_edge_err = 0.0
        num_terms = 0.0
        if is_ensemble
            for k in model.simulation_indices
                cell_L = model.cell_sol[k].u[j][end]
                pde_L = sol.u[j][end]
                edge_loss = (cell_L - pde_L)^2
                if use_relative_err
                    edge_loss /= cell_L^2
                end
                leading_edge_err += edge_loss
                num_terms += 1
            end
        else
            cell_L = model.cell_sol.u[j][end]
            pde_L = sol.u[j][end]
            edge_loss = (cell_L - pde_L)^2
            if use_relative_err
                edge_loss /= cell_L^2
            end
            leading_edge_err += edge_loss
            num_terms += 1
        end
        return leading_edge_err, num_terms
    else
        return 0.0, 1
    end
end

@inline function evaluate_density_loss(model, θ_train, test_subset, indicators=model.indicators; use_relative_err=true, leading_edge_error=true, conserve_mass=false)
    try
        pde_sol = solve_subproblem_pde(model, θ_train, indicators, conserve_mass)
        !SciMLBase.successful_retcode(pde_sol) && return Inf
        if model.pde_template isa MBProblem
            @views any(≤(0), pde_sol[end, :]) && return Inf
        end
        @floop FLOOPS_EX for j in test_subset
            density_err, num_terms = evaluate_density_error(pde_sol, model, j; use_relative_err)
            @reduce(density_loss = 0.0 + density_err)
            @reduce(density_summand = 0 + num_terms)
            if model.pde_template isa MBProblem
                leading_edge_err, num_terms = evaluate_leading_edge_error(pde_sol, model, j; use_relative_err, leading_edge_error)
                @reduce(leading_edge_loss = 0.0 + leading_edge_err)
                @reduce(leading_edge_summand = 0 + num_terms)
            end
        end
        if model.pde_template isa MBProblem
            loss = log(density_loss / density_summand)
            if leading_edge_error
                loss += log(leading_edge_loss / leading_edge_summand)
            end
        else
            loss = log(density_loss / density_summand)
        end
        return loss
    catch e
        println(e)
        return Inf
    end
end