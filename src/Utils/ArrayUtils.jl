

GetOrder(X::AbstractArray) = length(size(X))

NbElements(x::AbstractArray) = prod(size(x))
NbElements(x::Number) = 1

dropdims(X) = dropdims(X, dims=Tuple(findall(size(X) .== 1)))

function OneHot(loc::Int, size=10)
    Y = zeros(Float64, size, 1)
    Y[loc] = 1
    return Y
end

function SameOrder(X::AbstractArray, Y::AbstractArray)
    diff = length(size(X)) - length(size(Y))
    if diff < 0 
        X = reshape(X, (size(X)..., ones(Int64, abs(diff))...)...)
    elseif diff > 0
        Y = reshape(Y, (size(Y)..., ones(Int64, abs(diff))...)...)
    end
    return X, Y
end
SameOrder(X::Number, Y::Number) = (X, Y)
SameOrder(X::Number, Y) = SameOrder([X], Y)
SameOrder(X, Y::Number) = SameOrder(X, [Y])

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