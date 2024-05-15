"""
    Dense
Dense layers consist of a linear mapping followed by 
a non linear element-wise activation function.
"""
Identity(X) = X
mutable struct Dense <: Mutator
    W
    B
    acf::Function

    function Dense(InDim::Int, OutDim::Int; acf=Identity)
        std = sqrt(2 / InDim) 
        W = randn(OutDim, InDim) * std
        B = zeros(OutDim, 1)
        return new(W, B, acf)
    end
end



(D::Dense)(X) = D.acf(D.W * X + D.B)

function AddParams!(ctx::⬅Context, D::Dense)
    D.W = AddParams!(ctx, D.W)
    D.B = AddParams!(ctx, D.B)
end

function Untrack!(D::Dense)
    D.W = D.W.val
    D.B = D.B.val
end
