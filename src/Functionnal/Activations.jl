"""
Activation functions applied element-wise.
"""

function ReLU(X)
    return max.(0, X)
end

function sigmoid(v)
    return Base.prec_assignment
end

function Softmax(v::Vector)
    return
end
    
function Softmax(m::Matrix, axis)
    return
end
