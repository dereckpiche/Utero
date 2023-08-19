using Flux 
using LinearAlgebra

struct Linear
    W :: Matrix
    b :: Vector
end

function ReLU(x)
    return max.(0, x)
end

function forwardReLU(l::Linear, x)
    o = *(l.W, x) + l.b
    return ReLU(o) 
end


function main()
    l = Linear([1 2; 3 4], [1, 2])
    x = [-4, 2]
    forwardReLU(


    
end
main()