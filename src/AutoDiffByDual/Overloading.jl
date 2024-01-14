mutable struct ⬅Context
    Params
    Tape
    Gradients::Dict
    Counter
    function ⬅Context()
        return new([], [], Dict(), [0])
    end
end


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


mutable struct ⬅Tracker{T}
    val::T
    id::Int64 # Static identification function
    linkers
    parents
    function ⬅Tracker(val) 
        global Counter[1] += 1
        id = Counter[1]
        return new{typeof(val)}(val, id, [], [])
    end

    function ⬅Tracker(ctx::⬅Context, val) 
        ctx.Counter[1] += 1
        return new{typeof(val)}(val, ctx.Counter[1], [], [])
    end
end


⬅Overloaded = (:+, :-, :*, :/, :^, :sin, :cos)
for f in ⬅Overloaded
    @eval begin
    function Base.$f(x::⬅Tracker, y::⬅Tracker)
        z, Linker = ⬅Dual($f, x.val, y.val)
        z = ⬅Tracker(z)
        for (s, i) in [(x, 1), (y, 2)]
            push!(s.linkers, ∇ -> Linker(∇)[i])
            push!(s.parents, z.id)
            push!(Tape, s)
        end
        return z
    end

    @eval begin
        function Base.$f(x::⬅Tracker)
            (z, Linker) = ⬅Dual($f, x.val)
            z = ⬅Tracker(z)
            push!(x.linkers, ∇ -> Linker(∇))
            push!(x.parents, z.id)
            push!(Tape, x)
            return z
        end
    end

    Base.$f(x::⬅Tracker, y::Real) = Base.$f(x, ⬅Tracker(y))
    Base.$f(x::Real, y::⬅Tracker) = Base.$f(⬅Tracker(x), y)
end
end

"""
convert(::Type{⬅Tracker}, x::Real) = ⬅Tracker(x)
convert(::Type{⬅Tracker}, x::Int) = ⬅Tracker(x)
promote_rule(::Type{⬅Tracker}, ::Type{<:Number}) = ⬅Tracker
"""