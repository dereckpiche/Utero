function GenNode(Nodes)
    # generate a new Node (Nodeentification in the graph)
    Node = convert(Int64, floor(10000000 * rand()))
    while Node in Nodes
        Node = convert(Int64, floor(10000000 * rand()))
    end
    Nodes = union!(Nodes, Node)
    return Node
end

mutable struct Tracked{T} <: Real 
    val::T
    Node::Int64 # identification in the computationnal graph
    Tracked(val, Nodes) = return new{typeof(val)}(val, GenNode(Nodes))
    Tracked(val, Node) = return new{typeof(val)}(val, Node) 
end
