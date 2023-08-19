"""
Loss functions.
"""

function mse(v1::Vector, v2::Vector)
    mse = 0
    l = length(v1)
    for i in 1:l
        mse += (v1[i] - v2[i])^2
    end
    return mse / l
end

