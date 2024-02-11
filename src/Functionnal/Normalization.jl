Normalize(X, Total=1) =  X * (Total / sum(X))

function Softmax(X, Temp=1)
    expo = exp.(Temp .* X)
    expsum = sum(expo, dims=1)
    return expo ./ expsum
end

