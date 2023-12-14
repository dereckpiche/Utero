"""
Activation functions applied element-wise.
"""

function ReLU(v)
    return max.(0, v)
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
