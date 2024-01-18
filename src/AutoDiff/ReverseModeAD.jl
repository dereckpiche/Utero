function Params(ctx::⬅Context, x::Real)
    x = ⬅Tracker(ctx, x)
    push!(ctx.Params, x.id)
    return x
end

function Params(ctx::⬅Context, X...)
    ps = []
    for x in X 
        x = ⬅Tracker(ctx, x)
        push!(ctx.Params, x.id)
        push!(ps, x)
    end
    return ps
end



function Jacobians(ctx::⬅Context)
    grads = []
    for p in ctx.Params 
        push!(grads, ctx.Jacobians[p])
    end
    return grads
end

function ⬅ChainStep(ctx::⬅Context, x::⬅Tracker)
    g = 0.0
    if !isempty(x.Chainers)
        l = pop!(x.Chainers)
        p = pop!(x.Childs)
        g = l(getindex(ctx.Jacobians, p))
    end
    if haskey(ctx.Jacobians, x.id)
        ctx.Jacobians[x.id] += g
    else 
        setindex!(ctx.Jacobians, g, x.id) 
    end 
    return g
end

function ForwardBackward(ctx::⬅Context, f::Function, X...)
    global Tape = ctx.Tape
    global Counter = ctx.Counter

    # Forward Pass
    y = f(X...)

    # Backward Pass
    setindex!(ctx.Jacobians, 1.0, y.id)
    push!(ctx.Tape, y)
    for z in reverse(ctx.Tape)
        ⬅ChainStep(ctx, z)
    end
    return (y.val, Jacobians(ctx))
end