# TODO: make this all clean and abstract once it works

struct ⬅Ctx
    Tape
    Gradients::Dict
    ⬅Ctx(tape) = new(tape, Dict())
end

mutable struct Tracker{T}
    val::T
    idf # Static identification function
    linkers
    parents
    function Tracker(val) 
        idf = _ -> nothing # easy way to create reference
        idf = rand() * 100
        t = new{typeof(val)}(val, idf, [], [])
        return t
    end
end

function ⬅grad(x::Tracker)
    g = 0.0
    for (l, p) in zip(x.linkers, x.parents)
        g += l(getindex(⬅ctx.Gradients, p))
    end
    if haskey(⬅ctx.Gradients, x.idf)
        ⬅ctx.Gradients[x.idf] += g
    else 
        setindex!(⬅ctx.Gradients, g, x.idf) 
    end 
    return g
end


DualedFs = (:+, :-, :*, :/, :^, :sin, :cos)
for f in DualedFs
    @eval begin
    function Base.$f(x::Tracker, y::Tracker)
        z, Linker = ⬅Dual($f, x.val, y.val)
        z = Tracker(z)
        for (s, i) in [(x, 1), (y, 2)]
            push!(s.linkers, ∇ -> Linker(∇)[i])
            push!(s.parents, z.idf)
            push!(⬅ctx.Tape, s)
        end
        return z
    end

    @eval begin
        function Base.$f(x::Tracker)
            (z, Linker) = ⬅Dual($f, x.val)
            z = Tracker(z)
            push!(x.linkers, ∇ -> Linker(∇))
            push!(x.parents, z.idf)
            push!(⬅ctx.Tape, x)
            return z
        end
    end

    Base.$f(x::Tracker, y::Real) = Base.$f(x, Tracker(y))
    Base.$f(x::Real, y::Tracker) = Base.$f(Tracker(x), y)
end
end

"""
convert(::Type{Tracker}, x::Real) = Tracker(x)
convert(::Type{Tracker}, x::Int) = Tracker(x)
promote_rule(::Type{Tracker}, ::Type{<:Number}) = Tracker
"""