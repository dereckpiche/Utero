function Normalize(X, Total=1)
    InitialTotal = sum(X)   
    return X * (Total / InitialTotal)
end

function Softmax(X, Temp=1)
    exp = exp.(Temp .* X)
    expsum = sum(exp)
    return exp ./ expsum
end

