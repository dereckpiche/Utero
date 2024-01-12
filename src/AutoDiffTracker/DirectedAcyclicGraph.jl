function AddEdge(Edges, source, sink)
    if source in keys(Edges) 
        merge!(Edges, IdDict(source=>Set([sink, get(Edges, source, false)...])))
    else 
        merge!(Edges, IdDict(source=>Set([sink]))) 
    end 
end

function RemoveEdge(Edges, source, sink)
    if source in keys(Edges) delete!(get(Edges, source, false), sink) end
end

function ReverseEdges(Edges)::IdDict
    ReversedEdges = IdDict{}
    for (source, sinks) in edges
        for sink in sinks 
            AddEdge(ReversedEdges, source, sink)
        end
    end
    return ReversedEdges
end

function HasIncomingEdge(Edges, Node)::Bool
    for sinks in values(Edges)  
        if (Node in sinks) return true end 
    end
    return false
end


function KahnTopoSort(Nodes::Set, edges::IdDict)
    # Kahn's topological sorting algorithm
    # edges is a DAG in the form of a dictionnary (source => sinks)
    Edges = deepcopy(edges)
    Sorted = []
    NoIncomingEdges = Set( [Node for Node in Nodes  if !HasIncomingEdge(Edges, Node)] )
    while !isempty(NoIncomingEdges)
        source = pop!(NoIncomingEdges)
        Sorted = cat(Sorted, [source], dims=1)
        sinks = get(Edges, source, false)
        if sinks!=false for sink in sinks
            RemoveEdge(Edges, source, sink)
            if !HasIncomingEdge(Edges, sink) 
                NoIncomingEdges = union(NoIncomingEdges, Set(sink)) 
            end
        end 
    end
    return Sorted
end


Nodes = Set([1, 2, 3, 4, 5])
Edges = IdDict(1=>Set([2, 3]), 4=>Set([5]), 2=>Set([4, 5]), 3=>Set([4]))
@show KahnTopoSort(Nodes, Edges) #should return [5, 4, 2, 3, 1] or [5, 2, 4, 3, 1]