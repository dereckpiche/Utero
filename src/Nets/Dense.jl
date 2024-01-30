"""
    Dense
Dense layers consist of a linear mapping followed by 
a non linear element-wise activation function.
"""
mutable struct Dense 
    W
    B
    activation::Function

    function Dense(InDim::Int, OutDim::Int, activation::Function)
        W = randn(OutDim, InDim)
        B = randn(OutDim, 1)
        return new(W, B, activation)
    end
end

(D::Dense)(X) = D.activation(D.W * X + D.B)

function AddParams!(ctx::⬅Context, D::Dense)
    D.W = AddParams!(ctx, D.W)
    D.B = AddParams!(ctx, D.B)
end
