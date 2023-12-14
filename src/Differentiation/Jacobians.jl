

mutable struct Tracked{T} <: Real
    val::T
    Node::Int64 # identification in the computationnal graph
    Tracked(val, Nodes) = return new{typeof(val)}(val, GenNode(Nodes))
    Tracked(val, Node) = return new{typeof(val)}(val, Node)
end


for t in (Symbol(Integer), Symbol(AbstractFloat))
    eval(
        quote
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

            GetJacobian(f::typeof(sin), a::Tracked) = cos(a.val)
            GetJacobian(f::typeof(cos), a::Tracked) = -sin(a.val)
        end
    )
end