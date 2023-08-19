"""
Loss functions.
"""

function mse(v1::Vector, v2::Vector)
    """
    Mean error squared.
    """
    mse = 0
    l = length(v1)
    for i in 1:l
        mse += (v1[i] - v2[i])^2
    end
    return mse / l
end

function mae(v1::Vector, v2::Vector)
    """
    Mean absolute error.
    """
    mae = 0
    l = length(v1)
    for i in 1:l
        mae += abs(v1[i] - v2[i]) 
    end
    return msa / l
end