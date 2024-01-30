"""
    Activation.jl
Activation functions applied element-wise.
"""

function ReLU(X)
    return max.(0, X)
end

Sigmoid(x::Real) = 1 / (1 + exp(1)^(-x))

function Sigmoid(X::AbstractArray)
    return map(x -> 1 / (1 + exp(1)^(-x)), X)
end

    

