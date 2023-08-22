# Solve Ax = b, constrained so that x[i] = 0 if indicators[i] == false
function projected_solve(A, b, indicators)
    x = zeros(size(A, 2)) 
    x[indicators] .= A[:, indicators] \ b # We could do @views here, but not having it is sometimes faster due to the size of our matrices
    return x
end
function projected_solve(model::EQLModel, training_subset, indicators=model.indicators)
    return projected_solve(model.A[training_subset, :], model.b[training_subset], indicators)
end

@inline function evaluate_regression_loss(model::EQLModel{S,P}, θ_train, test_subset, indicators=model.indicators; use_relative_err=true) where {S,P}
    A_test = model.A[test_subset, indicators]
    b_test = model.b[test_subset]
    regression_loss = norm(A_test * θ_train[indicators] - b_test)^2
    if use_relative_err
        regression_loss /= norm(b_test)^2
    end
    return log(regression_loss::Float64 / length(test_subset))
end