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
for  f in DualedFs
    @eval begin
        function Base.$f(x::Tracker, y::Tracker)
            (z, Linker) = ⬅Dual(typeof($f), x.val, y.val)
            z = Tracker(z)
            λ = ∇ -> Linker(∇)[i]
            #@eval begin
            #    global ⬅grad(::typeof(z.idf), ::typeof(x.idf), ∇) = Linker(∇)[1]
            #end
            append!(x.parfs, z)
            #@eval begin
            #    global ⬅grad(::typeof(z.idf), ::typeof(y.idf), ∇) = Linker(∇)[2]
            #end
            append!(y.parfs, z)
            # add to Tape
            append!(⬅ctx.Tape, x, y)
            return z
        end
    end
end

convert(::Type{Tracker}, x::Real) = Tracker(x)
convert(::Type{Tracker}, x::Int) = Tracker(x)
promote_rule(::Type{Tracker}, ::Type{<:Number}) = Tracker