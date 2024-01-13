function CumulGrads(Gradients, GradIDTape)
    grads = Dict()
    for (i, ID) in enumerate(GradIDTape) 
        if ID == NotParam() continue end
        if haskey(grads, ID)
            grads[ID] += Gradients[i]
        else setindex!(grads, Gradients[i], ID) end
    end
    return grads
end


function ForwardBackward(f, x)
    global Gradients = [1.0]
    global Linkers = []
    global GradIDTape = []

    # Forward Pass
    l = f(x)

    # Backward Pass
    for (linker, ∂l∂z) in zip(Linkers, Gradients)
        append!(Gradients, linker(∂l∂z)...)
    end

    @show Gradients
    @show GradIDTape
    popfirst!(Gradients)
    return CumulGrads(reverse(Gradients), reverse(GradIDTape))
end