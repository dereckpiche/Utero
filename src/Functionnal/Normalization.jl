function Normalize(X, Total=1)
    InitialTotal = sum(X)   
    return X * (Total / InitialTotal)
end

function Softmax(X, Temp=1)
    sum = exp.(Temp.*X) .^ (1/2)
    sum = sum' * sum
    return X / sum
end