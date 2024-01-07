# Explicitely import to modify
import Base.:+
import Base.:-
import Base.:*
import Base.:/
import Base.:^
import Base.sin
import Base.cos


function AddJacobian(Jacobians, source, sink, Jacobian)
    # adds jacobian of sink with respect to source to jacobians
    merge!(Jacobians, IdDict((source, sink) => Jacobian)) 
end

GetJacobian(f::typeof(sin), a::Tracked) = cos(a.val)
GetJacobian(f::typeof(cos), a::Tracked) = -sin(a.val)

for t in (Symbol(Integer), Symbol(AbstractFloat))
    eval(
        quote
            # Scalar operations
            GetJacobian(f::typeof(+), a::($t), b::Tracked) = 1
            GetJacobian(f::typeof(+), a::Tracked, b::($t)) = 1

            GetJacobian(f::typeof(-), a::($t), b::Tracked) = -1
            GetJacobian(f::typeof(-), a::Tracked, b::($t)) = 1

            GetJacobian(f::typeof(*), a::($t), b::Tracked) = a
            GetJacobian(f::typeof(*), a::Tracked, b::($t)) = b

            GetJacobian(f::typeof(/), a::($t), b::Tracked) = (-1) / b.val^2
            GetJacobian(f::typeof(/), a::Tracked, b::($t)) = 1 / b

            GetJacobian(f::typeof(^), a::Tracked, b::($t)) = b * a.val^(b - 1)
            GetJacobian(f::typeof(^), a::($t), b::Tracked) = a^(b.val) * log(a)
        end
    )
end


GetJacobian(f::typeof(.+), a::Array, b::Tracked) = Diagonal(ones(prod(size(a)))) 
GetJacobian(f::typeof(.+), a::Tracked, b::Array) = Diagonal(ones(prod(size(a)))) 

GetJacobian(f::typeof(.-), a::Array, b::Tracked) = -Diagonal(ones(prod(size(a)))) 
GetJacobian(f::typeof(.-), a::Tracked, b::Array) = Diagonal(ones(prod(size(a)))) 

#GetJacobian(f::typeof(.*), a::Array, b::Tracked) = Diagonal(reshape(a, 1, :)) 
#GetJacobian(f::typeof(.*), a::Tracked, b::Array) = b

#GetJacobian(f::typeof(./), a::Array, b::Tracked) = @. (-1) / b.val^2
#GetJacobian(f::typeof(./), a::Tracked, b::Array) = @. 1 / b

#GetJacobian(f::typeof(.^), a::Tracked, b::Array) = @. b * a.val^(b - 1)
#GetJacobian(f::typeof(.^), a::Array, b::Tracked) = @. a^(b.val) * log(a)


function GetJacobian(f::typeof(*), w::Tracked, x::Array)
    J = Diagonal( reshape(x, prod(size(x))))
    return J
end
