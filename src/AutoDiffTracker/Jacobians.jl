# Explicitely import to modify
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



# Jacobians For Two Scalar Tensors Operations

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

# Jacobians For Single Scalar Tensor Operations

GetJacobian(f::typeof(sin), a::Tracked) = cos(a.val)
GetJacobian(f::typeof(cos), a::Tracked) = -sin(a.val)


# Jacobians For Two Tensor Operations

function GetJacobian(f::typeof(*), w::Tracked, x::AbstractMatrix) # Matrix Multiplication Jacobian
    m_w, n_w = size(w.val)
    m_x, n_x = size(x)
    J = spzeros(Float64, m_w * n_x, m_w * n_w)
    for i in 1:m_w
        J[i, (i-1)*n_w+1:(i)*n_w] = x[1:m_x] 
    end
    return J
end

function GetJacobian(f::typeof(*), w::AbstractMatrix, x::Tracked)
    m_w, n_w = size(w)
    m_x, n_x = size(x.val)
    J = spzeros(Float64, m_w * n_x, m_x * n_x)
    return J
end

GetJacobian(f::typeof(.+), a::Array, b::Tracked) = Diagonal(ones(prod(size(a)))) 
GetJacobian(f::typeof(.+), a::Tracked, b::Array) = Diagonal(ones(prod(size(a)))) 

GetJacobian(f::typeof(.-), a::Array, b::Tracked) = -Diagonal(ones(prod(size(a)))) 
GetJacobian(f::typeof(.-), a::Tracked, b::Array) = Diagonal(ones(prod(size(a)))) 

"""
GetJacobian(f::typeof(.*), a::Array, b::Tracked) = Diagonal(reshape(a, 1, :)) 
GetJacobian(f::typeof(.*), a::Tracked, b::Array) = b

GetJacobian(f::typeof(./), a::Array, b::Tracked) = @. (-1) / b.val^2
GetJacobian(f::typeof(./), a::Tracked, b::Array) = @. 1 / b

GetJacobian(f::typeof(.^), a::Tracked, b::Array) = @. b * a.val^(b - 1)
GetJacobian(f::typeof(.^), a::Array, b::Tracked) = @. a^(b.val) * log(a)
"""

# Jacobians For Single Tensor Operations

function GetJacobian(f::typeof(prod), w::Tracked)
    n = size(w.val)[2]
    p = prod(w.val)
    J = zeros(1,prod(size(w.val)))
    for i in 1:prod(size(w.val))
        s = w.val[Int64(i // n), i % n + 1]
        J[i] = p / s
    end
    return J
end

function GetJacobian(f::typeof(prod), w::Tracked, dims)
    #TODO
end
