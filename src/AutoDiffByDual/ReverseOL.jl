mutable struct ⬅Context
    Params
    Tape
    Jacobians::Dict
    Counter
    function ⬅Context()
        return new([], [], Dict(), [0])
    end
end



mutable struct ⬅Tracker{T}
    val::T
    id::Int64 
    Chainers
    Childs
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

"""
Overload primitive functions. Record the function order on tape.
"""

for f in ⬅Dualed
    @eval begin
    function $f(x::⬅Tracker, y::⬅Tracker)
        z, Chainer = ⬅Dual($f, x.val, y.val)
        z = ⬅Tracker(z)
        for (s, i) in [(x, 1), (y, 2)]
            push!(s.Chainers, ∇ -> Chainer(∇)[i])
            push!(s.Childs, z.id)
            push!(Tape, s)
        end
        return z
    end

    @eval begin
        function Base.$f(x::⬅Tracker)
            (z, Chainer) = ⬅Dual($f, x.val)
            z = ⬅Tracker(z)
            push!(x.Chainers, ∇ -> Chainer(∇))
            push!(x.Childs, z.id)
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