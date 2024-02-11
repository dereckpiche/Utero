function Normalize(X, Total=1)
    InitialTotal = sum(X)   
    return X * (Total / InitialTotal)
end

function Softmax(X, Temp=1)
    expo = exp.(Temp .* X)
    expsum = sum(expo, dims=1)
    return expo ./ expsum
end

