# TODO: make this all clean and abstract once it works

struct Param
    val
    ID
end

struct OvWrapper <:Number
    val
end

struct NotParam end

Linkers = []
GradIDTape = []

OvWrap(x::Real) = OvWrapper(x)
OvWrap(x::OvWrapper) = x

for  f in (:+, :-, :*, :/, :^)
    @eval begin
        function Base.$f(x::Param, y::Number)
            push!(GradIDTape, x.ID, NotParam())
            return $f(OvWrap(x.val), OvWrap(y))
        end
    end

    @eval begin
        function Base.$f(x::Number, y::Param)
            push!(GradIDTape, NotParam(), y.ID)
            return $f(OvWrap(x), OvWrap(y.val))
        end
    end

    @eval begin
        function Base.$f(x::Param, y::Param)
            push!(GradIDTape, x.ID, y.ID)
            return $f(OvWrap(x.val), OvWrap(y.val))
        end
    end
end



for  f in (:+, :-, :*, :/, :^)
    @eval begin
        function Base.$f(x::OvWrapper, y::OvWrapper)
            z, linker = ⬅Dual($f, x.val, y.val)
            push!(Linkers, linker)
            #pushfirst!(GradIDTape, NotParam())
            return OvWrap(z)
        end
    end
end

for  f in (:sin, :cos)
    @eval begin
        function Base.$f(x::Param)
            push!(GradIDTape, x.ID)
            return $f(OvWrap(x.val))
        end
    end
end

for  f in (:sin, :cos)
    @eval begin
        function Base.$f(x::OvWrapper)
            z, linker = ⬅Dual($f, x.val)
            push!(Linkers, linker)
            #pushfirst!(GradIDTape, NotParam())
            return OvWrap(z)
        end
    end
end

convert(::Type{OvWrapper}, x::Real) = OvWrapper(x)
promote_rule(::Type{OvWrapper}, ::Type{<:Number}) = OvWrapper