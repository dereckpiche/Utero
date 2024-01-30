"""
TODO: add "!" for side effects
"""
mutable struct ⬅Context
    Params
    Tape
    Gradients::Dict
    Counter
    function ⬅Context()
        return new([], [], Dict(), [0])
    end
end

function ⬅CleanTape!(ctx)
    ctx.Tape = []
end

function AddParams!(ctx::⬅Context, x::Real)
    x = ⬅Tracker(x)
    push!(ctx.Params, x)
    return x
end

function AddParams!(ctx::⬅Context, X...)
    ps = []
    for x in X 
        x = ⬅Tracker(x)
        push!(ctx.Params, x)
        push!(ps, x)
    end
    return ps
end

function PluckParamGrads(ctx::⬅Context)
    grads = []
    for p in ctx.Params
        push!(grads, ctx.Gradients[p.id])
    end
    return grads
end

function CumulChains!(ctx::⬅Context, x::⬅Tracker)
    g = 0.0
    if !isempty(x.Chainers)
        l = pop!(x.Chainers)
        p = pop!(x.Childs)
        g = l(get(ctx.Gradients, p, 0))
    end
    if haskey(ctx.Gradients, x.id)
        ctx.Gradients[x.id] += g
    else 
        setindex!(ctx.Gradients, g, x.id) 
    end 
    return g
end

function ForwardBackward!(ctx::⬅Context, f::Function, X...)
    global Tape = ctx.Tape
    global Counter = ctx.Counter

    # Forward Pass

    y = f(X...)

    # Backward Pass
    setindex!(ctx.Gradients, 1.0, y.id)
    push!(ctx.Tape, y)
    for z in reverse(ctx.Tape)
        CumulChains!(ctx, z)
    end
    ⬅CleanTape!(ctx)
    return (y.val, PluckParamGrads(ctx))
end


