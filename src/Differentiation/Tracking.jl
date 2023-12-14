function Track(Nodes, Tensor)
    Tensor = Tracked(Tensor, Nodes)
end

function Track(Nodes, d::Dense)
    Track(d.W)
    Track(d.B)
    d.B = Tracked(B, Vector)
end


mutable struct Tracked{T} <: Real
    val::T
    Node::Int64 # identification in the computationnal graph
    Tracked(val, Nodes) = return new{typeof(val)}(val, GenNode(Nodes))
    Tracked(val, Node) = return new{typeof(val)}(val, Node)
end

