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

function (d::dense)(x::Array{Real, 2})
    # apply dense to each row
    y = Array{Real, 2}(size(x)[0], size(d.w)[0])
    for (rowInd, v) in enumerate(eachrow(x))
        y[rowInd] = d(v)
    end
    return y
end
