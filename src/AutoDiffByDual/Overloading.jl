struct Param
    val
    ID
end


struct OvWrapper <:Number
    val
end

struct OvWrapper <:Number
    val
end

Linkers = []
TapeIDS = []

for  f in (:+, :-, :*, :/, :^)
    @eval begin
        function Base.$f(x::Tracker, y::Real)
            return $f(OvWrapper(x.val), OvWrapper(y))
        end
    end

    @eval begin
        function Base.$f(x::Real, y::Tracker)
            return $f(OvWrapper(x), OvWrapper(y.val))
        end
    end

    @eval begin
        function Base.$f(x::Tracker, y::Tracker)
            return $f(OvWrapper(x), OvWrapper(y.val))
        end
    end
end



for  f in (:+, :-, :*, :/, :^)
    @eval begin
        function Base.$f(x::OvWrapper, y::OvWrapper)
            z, linker = ⬅Dual($f, x.val, y.val)
            pushfirst!(Linkers, linker)
            #pushfirst!(argsss, 2) #TODO
            return Tracker(z)
        end
    end
end

for  f in (:sin, :cos)
    @eval begin
        function Base.$f(x::OvWrapper)
            z, linker = ⬅Dual($f, x.val)
            pushfirst!(Linkers, linker)
            return OvWrapper(z)
        end
    end
end

convert(::Type{OvWrapper}, x::Real) = OvWrapper(x)
promote_rule(::Type{OvWrapper}, ::Type{<:Number}) = OvWrapper