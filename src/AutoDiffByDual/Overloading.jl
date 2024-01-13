# TODO: make this all clean and abstract once it works

struct ⬅Ctx
    Tape
    ⬅Ctx(tape) = new(tape)
end

struct Tracker{T}
    val::T
    idf # Static identification function
    parfs 
    function Tracker(val) 
        idf = _ -> nothing # easy way to create reference
        return new{typeof(val)}(val, idf, [])
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
    function Base.$f(X::Tracker...)
        (z, Linker) = ⬅Dual($f, [x.val for x in X]...)
        z = Tracker(z)
        λ = ∇ -> Linker(∇)[i]
        for (i, x) in enumerate(X)
            @eval begin
            function ⬅grad(::$zidf, ::$zidf, ∇) Linker(∇)[i] end
            end
            append!(x.parfs, z)
            append!(⬅ctx.Tape, x, y)
        end
        return z
    end
    Base.$f(x::Tracker, y::Real) = Base.$f(x, Tracker(y))
    Base.$f(x::Real, y::Tracker) = Base.$f(Tracker(x), u)
end
end

convert(::Type{Tracker}, x::Real) = Tracker(x)
convert(::Type{Tracker}, x::Int) = Tracker(x)
promote_rule(::Type{Tracker}, ::Type{<:Number}) = Tracker