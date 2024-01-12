

function ⬅ChainRule(∂l∂z, ids, linker)
    sum = 0
    ∂ls = linker(∂l∂z)
    for (id, ∂l) in zip(ids, ∂ls)
         sum += id*∂l 
    end
    return sum
end


function ForwardBackward(f, x)
    global Gradients = [1.0]
    global Linkers = [] 

    # Forward Pass
    l = f(x)

    for l in Linkers println(l) end

    # Backward Pass
    for (linker, ∂l∂z) in zip(Linkers, Gradients)
        append!(Gradients, linker(∂l∂z)...)
    end
    return Gradients
end