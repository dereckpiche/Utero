"""
Loss functions.
"""

""" 
    mse
Mean error squared of two vectors.
"""

function mse(v1::Vector, v2::Vector)
    diff = v1 - v2
    return sum(diff .^ 2) / size(v1)[1]
end

mse(m1::Array, m2::Array) = mse(vec(m1), vec(m2))

""" 
Mean absolute error
"""
function mae(v1::Vector, v2::Vector)
    mae = 0
    l = length(v1)
    for i in 1:l
        mae += abs(v1[i] - v2[i]) 
    end
    return msa / l
end