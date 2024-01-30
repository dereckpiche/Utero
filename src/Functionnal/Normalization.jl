function Normalize(X::AbstractArray, Total=1)
    InitialTotal = sum(X)   
    return X * (Total / InitialTotal)
end

function Softmax(X::AbstractArray, Temp=1)
    #TODO
end