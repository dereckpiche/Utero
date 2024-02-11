function MeanSquaredError(X, Y)
    D = X - Y
    return D' * D ./ length(X)
end

function CrossEntropy(X, Y)
    
end