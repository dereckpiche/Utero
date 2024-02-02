function Normalize(X, Total=1)
    InitialTotal = sum(X)   
    return X * (Total / InitialTotal)
end

function Softmax(X, Temp=1)
    #sum = Temp .* X
    #sum = exp.(sum)
    sum = X' * X
    #sum = sum .^ (1/2)
    return sum .* X 
end

