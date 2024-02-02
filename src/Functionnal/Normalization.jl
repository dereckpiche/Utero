function Normalize(X, Total=1)
    InitialTotal = sum(X)   
    return X * (Total / InitialTotal)
end

function Softmax(X, Temp=1)
    sum = Temp.*X
    sum = exp.(sum)
    sum = sum .^ (1/2)
    sum = sum' * sum
    return X / sum
end