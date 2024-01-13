# TODO: make this all clean and abstract once it works

struct ⬅Ctx
    Tape
    ⬅Ctx(tape) = new(tape)
end

mutable struct Tracker
    val
    idf # Static identification function
    parfs 
    function Tracker(val) 
        idf = _ -> nothing # easy way to create reference
        return new(val, idf, [])
    end
end

"""
function ⬅grad(x::Tracker)
    g = 0.0
    for partial in x.parfs
        g += partial()
    end
end
"""


DualedFs = (:+, :-, :*, :/, :^, :sin, :cos)
for f in DualedFs
    @eval begin
    global function Base.$f(X::Tracker...)
        (z, Linker) = ⬅Dual(typeof($f), [x.val for x in X]...)
        z = Tracker(z)
        λ = ∇ -> Linker(∇)[i]
        for (i, x) in enumerate(X)
            @eval begin
                global ⬅grad(::typeof(z.idf), ::typeof(x.idf), ∇) = Linker(∇)[i]
            end
            append!(x.parfs, z)
            append!(⬅ctx.Tape, x, y)
        end
        return z
    end
end
end

convert(::Type{Tracker}, x::Real) = Tracker(x)
convert(::Type{Tracker}, x::Int) = Tracker(x)
promote_rule(::Type{Tracker}, ::Type{<:Number}) = Tracker