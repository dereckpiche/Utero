mutable struct dense
    w::Matrix
    b::Vector
    f::Function
end

function dense(inDim::Int, outDim::Int, f::Function)
    # returns dense composite according to dimensions
    w = rand(Float16, (outDim, inDim))
    b = rand(Float16, (outDim))
    return dense(w, b, f)
end


function (d::dense)(v::Vector)
    return d.f( *(d.w, v) + d.b )
end

