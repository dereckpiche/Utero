function Gradients(ctx::⬅Context)
    grads = []
    for p in ctx.Params 
        push!(grads, ctx.Gradients[p])
    end
    return grads
end

function ⬅Chain(ctx::⬅Context, x::⬅Tracker)
    g = 0.0
    if !isempty(x.linkers)
        l = pop!(x.linkers)
        p = pop!(x.parents)
        g = l(getindex(ctx.Gradients, p))
    end
    if haskey(ctx.Gradients, x.id)
        ctx.Gradients[x.id] += g
    else 
        setindex!(ctx.Gradients, g, x.id) 
    end 
    return g
end

function ForwardBackward(ctx::⬅Context, f::Function, x)
    global Tape = ctx.Tape
    global Counter = ctx.Counter

    # Forward Pass
    l = f(x)

    # Backward Pass
    setindex!(ctx.Gradients, 1.0, l.id)
    push!(ctx.Tape, l)
    for x in reverse(ctx.Tape)
        ⬅Chain(ctx, x)
    end
    return Gradients(ctx)
end