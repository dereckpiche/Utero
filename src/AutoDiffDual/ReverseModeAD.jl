import Base.:+
import Base.:-
import Base.:*
import Base.:/
import Base.:^
import Base.sin
import Base.cos
import Base.map
import Base.prod
import Base.sum

function ⬅ChainRule(∂l∂z, linker)
    sum = 0
    ∂ls = linker(∂l∂z)
    for ∂l in ∂ls[2:end] sum += ∂l end
    return sum
end

mutable struct Param{T}
    val::T
end

Chain = []

for  f in (:+, :-, :*, :/, :^)
@eval begin
    global function ($f)(x::T, y::Param) where {T<:Number}
        z, linker = ⬅Dual(($f), x, y.val)
        pushfirst!(Chain, linker)
        pushfirst!(argsss, 2) #TODO
        return z
    end

    global function ($f)(x::Param, y::T) where {T<:Number}
        z, linker = ⬅Dual(($f), x.val, y)
        pushfirst!(Chain, linker)
        return z
    end
end
end

for  f in (:sin, :cos)
    @eval begin
        global function ($f)(x::Param)
            z, linker = ⬅Dual(($f), x.val)
            pushfirst!(Chain, linker)
            return z
        end
    end
end



function ForwardBackward(f, x)
    global Chain = [] # reset the chain

    # Forward Pass
    l = Base.invokelatest(f, x)
    
    # Backward Pass
    ∂l∂z = 1
    for linker in Chain
        ∂l∂z = ⬅ChainRule(∂l∂z, linker)
    end
    return ∂l∂z
end