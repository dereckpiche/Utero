# TODO: make this all clean and abstract once it works

struct ⬅Ctx
    Tape
    ⬅Ctx(tape) = new(tape)
end

mutable struct Tracker{T}
    val::T
    grad
    idf # Static identification function
    linkers
    parents
    function Tracker(val) 
        idf = _ -> nothing # easy way to create reference
        return new{typeof(val)}(val, 0.0, idf, Set([]), [])
    end
end

function ⬅grad(x::Tracker)
    for (linker, parent) in zip(x.linkers, x.parents)
        x.grad += linker(parent.grad)
    end
    return x.grad
end


DualedFs = (:+, :-, :*, :/, :^, :sin, :cos)
for f in DualedFs
    @eval begin
    function Base.$f(X::Tracker...)
        (z, Linker) = ⬅Dual($f, [x.val for x in X]...)
        z = Tracker(z)
        for (i, x) in enumerate(X)
            push!(x.linkers, ∇ -> Linker(∇)[i])
            push!(x.parents, z)
            push!(⬅ctx.Tape, x)
        end
        return z
    end
    Base.$f(x::Tracker, y::Real) = Base.$f(x, Tracker(y))
    Base.$f(x::Real, y::Tracker) = Base.$f(Tracker(x), y)
end
end

convert(::Type{Tracker}, x::Real) = Tracker(x)
convert(::Type{Tracker}, x::Int) = Tracker(x)
promote_rule(::Type{Tracker}, ::Type{<:Number}) = Tracker