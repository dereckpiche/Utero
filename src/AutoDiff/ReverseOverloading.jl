mutable struct ⬅Tracker{T}
    val::T
    id::UUID
    Chainers
    Childs
    function ⬅Tracker(val) 
        return new{typeof(val)}(val, uuid1(), [], [])
    end
    ⬅Tracker(x::⬅Tracker) = x
end

#Base.length(X::⬅Tracker) = length(X.val)

function Untrack(X::⬅Tracker...)
    return [x.val for x in X]
end

HasTracker(args...) = any(isa(arg, ⬅Tracker) for arg in args)

"""
    ⬅Overload(func)
Inefficient way of overloading.
macro ⬅Overload(func)
    return :(
        function func(args...; kwargs...) 
            if HasTracker(args...)
                #TODO
            else
                #TODO
            end
        end
     )
end
"""

"""
    ⬅Overload(mode, func)
Efficient way of overloading.
"""

macro ⬅Overload(mode, func)
    # =================== Operators with single differentiable tensor
    if mode == :Unary
        return :(
            function $func(x::⬅Tracker, args...; kwargs...) 
                (z, Chainer) = ⬅Dual($func, x.val, args...; kwargs...)
                z = ⬅Tracker(z)
                push!(x.Chainers, ∇ -> Chainer(∇))
                push!(x.Childs, z.id)
                push!(Tape, x)
                return z
            end
         )
    
    # =================== Operators with two possible differentiable tensor
    elseif mode == :Binary
        return quote
        function $func(x::⬅Tracker, y::⬅Tracker, args...; kwargs...) 
            z, Chainer = ⬅Dual($func, x.val, y.val, args...; kwargs...)
            z = ⬅Tracker(z)
            for (i, s) in enumerate([x, y])
                push!(s.Chainers, ∇ -> Chainer(∇)[i])
                push!(s.Childs, z.id)
                push!(Tape, s)
            end
            return z
        end,

        function $func(x::⬅Tracker, y, args...; kwargs...) 
            z, Chainer = ⬅Dual($func, x.val, y, args...; kwargs...)
            z = ⬅Tracker(z)
            push!(x.Chainers, ∇ -> Chainer(∇)[1])
            push!(x.Childs, z.id)
            push!(Tape, x)
            return z
        end,

        function $func(x, y::⬅Tracker, args...; kwargs...) 
            z, Chainer = ⬅Dual($func, x, y.val, args...; kwargs...) 
            z = ⬅Tracker(z)
            push!(y.Chainers, ∇ -> Chainer(∇)[2])
            push!(y.Childs, z.id)
            push!(Tape, y)
            return z
        end
    end

    # =================== Operators with two possible differentiable tensor
    elseif mode == :BroadcastedBinary
        return :(
            function Base.broadcasted(::typeof($func),
                x::⬅Tracker, y::⬅Tracker, args...; kwargs...) 
                z, Chainer = ⬅Dual(Base.broadcasted, 
                $func, x.val, y.val, args...; kwargs...)
                z = ⬅Tracker(z)
                for (s, i) in [(x, 1), (y, 2)]
                    push!(s.Chainers, ∇ -> Chainer(∇)[i])
                    push!(s.Childs, z.id)
                    push!(Tape, s)
                end
                return z
            end,
    
            function Base.broadcasted(::typeof($func), 
                x::⬅Tracker, y, args...; kwargs...) 
                z, Chainer = ⬅Dual(Base.broadcasted, 
                $func, x.val, y, args...; kwargs...)
                z = ⬅Tracker(z)
                push!(x.Chainers, ∇ -> Chainer(∇)[1])
                push!(x.Childs, z.id)
                push!(Tape, x)
                return z
            end,
    
            function Base.broadcasted(::typeof($func), 
                x, y::⬅Tracker, args...; kwargs...) 
                z, Chainer = ⬅Dual(Base.broadcasted, 
                $func, x, y.val, args...; kwargs...)
                z = ⬅Tracker(z)
                push!(y.Chainers, ∇ -> Chainer(∇)[2])
                push!(y.Childs, z.id)
                push!(Tape, y)
                return z
            end
        )
    end
end        



""" Overloadings """

# ================================
# Element-Wise
# ================================

@⬅Overload Binary Base.:+

@⬅Overload Binary Base.:-

@⬅Overload Binary Base.:*
@⬅Overload BroadcastedBinary Base.:*

@⬅Overload Binary Base.:/
@⬅Overload BroadcastedBinary Base.:/

@⬅Overload Binary Base.:^
@⬅Overload BroadcastedUnary Base.:^

@⬅Overload Unary Base.exp
@⬅Overload BroadcastedUnary Base.exp

@⬅Overload Unary Base.sin

@⬅Overload Unary Base.cos

@⬅Overload Unary ReLU 

@⬅Overload Unary Sigmoid


# ================================
# Linear Algebra
# ================================

#@⬅Overload Binary Base.:*

@⬅Overload Unary Base.adjoint


# ================================
# Restructuring
# ================================
@⬅Overload Unary sum

@⬅Overload Unary getindex



