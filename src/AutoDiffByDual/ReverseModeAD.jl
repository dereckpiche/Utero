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
    global ⬅ctx = ⬅Ctx([])

    # Forward Pass
    l = f(x)
    # Backward Pass
    ⬅ctx.Tape[end].grad=1
    for x in ⬅ctx.Tape
        for y in ⬅ctx.Tape print([t.grad for t in y.parents]) end
        println()
        ⬅grad(x)
    end

end