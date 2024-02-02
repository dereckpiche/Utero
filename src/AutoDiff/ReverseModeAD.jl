"""
TODO: add "!" for side effects
"""
mutable struct ⬅Context
    Params
    Tape
    Gradients::Dict
    function ⬅Context()
        ctx = new([], [], Dict())
        global Tape = ctx.Tape
        return ctx
    end
end

function ⬅CleanContext!(ctx)
    ctx.Tape = []
    ctx.Gradients = Dict()
end

function AddParams!(ctx::⬅Context, x::Union{Real, AbstractArray})
    x = ⬅Tracker(x)
    push!(ctx.Params, x)
    return x
end

AddParams!(ctx::⬅Context, x::Function) = nothing

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
        ctx.Gradients[x.id] = ctx.Gradients[x.id] .+ g
    else 
        setindex!(ctx.Gradients, g, x.id) 
    end 
    return g
end

function ForwardBackward!(ctx::⬅Context, f::Function, X...)
    ⬅CleanContext!(ctx)
    global Tape = ctx.Tape

    # Forward Pass

    y = f(X...)
    # Backward Pass
    isa(y.val, Number) ? g = 1 : g = ones(size(y.val))
    setindex!(ctx.Gradients, g, y.id)
    push!(ctx.Tape, y)
    for z in reverse(ctx.Tape)
        CumulChains!(ctx, z)
    end
    paramgrads = PluckParamGrads(ctx)
    ⬅CleanContext!(ctx)
    return y.val, paramgrads
end


