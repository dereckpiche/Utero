

GetOrder(X::AbstractArray) = length(size(X))

NbElements(x::AbstractArray) = prod(size(x))

function OneHot(loc::Int, size=10)
    Y = zeros(Float64, size, 1)
    Y[loc] = 1
    return Y
end

function Commonize(X, Y)
    newX = X
    newY = Y
    if NbElements(X) > NbElements(Y)
        newY = fill(Y[1], size(X))
    else 
        newX = fill(X[1], size(Y))
    end

    return newX, newY
end