function MeanSquaredError(X, Y)
    D = X - Y
    return D' * D
end