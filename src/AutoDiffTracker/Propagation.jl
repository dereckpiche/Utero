
TwoTensorOperations = (
    Symbol(+), 
    Symbol(-), 
    Symbol(*), 
    Symbol(/), 
    Symbol(^)
)

SingleTensorOperations = (
    Symbol(map),
    Symbol(prod),
    Symbol(sum),
    Symbol(cos), 
    Symbol(sin)
)

function AddJacobian(Jacobians, source, sink, Jacobian)
    # adds jacobian of sink with respect to source to jacobians
    merge!(Jacobians, IdDict((source, sink) => Jacobian)) 
end


function ForwProp(f, x, ParameterIDs::Set)

    # Overcharge the operators to create a computationnal graph as well as 
    # the intermediate jacobians for backpropagation at a later stage

    global NodeIds = deepcopy(ParameterIDs)
    global Edges = IdDict()
    global Jacobians = IdDict()

    for op in TwoTensorOperations
        for t in (Symbol(Integer), Symbol(AbstractFloat), Symbol(Array))

            eval(:(global function ($op)(a::T, b::Tracked) where {T<:($t)}
                ID = AddNewID(NodeIds)
                J = GetJacobian(($op), a, b)
                AddEdge(Edges, b.ID, ID)
                AddJacobian(Jacobians, b.ID, ID, J)
                return Tracked(($op)(a, b.val), ID)
            end))

            eval(:(global function ($op)(a::Tracked, b::T) where {T<:($t)}
                ID = AddNewID(NodeIds)
                J = GetJacobian(($op), a, b)
                AddEdge(Edges, a.ID, ID)
                AddJacobian(Jacobians, a.ID, ID, J)
                return Tracked(($op)(a.val, b), ID)
            end))

            eval(:(global function ($op)(a::Tracked, b::Tracked)
                ID = AddNewID(NodeIds)
                Ja = GetJacobian(($op), a, b.val)
                AddEdge(Edges, a.ID, ID)
                AddJacobian(Jacobians, a.ID, ID, Ja)
                Jb = GetJacobian(($op), a.val, b)
                AddEdge(Edges, b.ID, ID)
                AddJacobian(Jacobians, b.ID, ID, Jb)
                return Tracked(($op)(a.val, b.val), ID)
            end))
        end
    end


    for op in SingleTensorOperations
        eval(
            :(
                global function ($op)(a::Tracked)
                    ID = AddNewID(NodeIds)
                    J = GetJacobian(($op), a)
                    AddEdge(Edges, a.ID, ID)
                    AddJacobian(Jacobians, a.ID, ID, J)
                    return Tracked(($op)(a.val), ID)
                end
            )
        )

        eval(
            :(
                global function ($op)(a::Tracked, args...)
                    ID = AddNewID(NodeIds)
                    J = GetJacobian(($op), a, args...)
                    AddEdge(Edges, a.ID, ID)
                    AddJacobian(Jacobians, a.ID, ID, J)
                    return Tracked(($op)(a.val, args...), ID)
                end
            )
        )
    end


    y = Base.invokelatest(f, x)
    return (y, NodeIds, Edges, Jacobians)
end



function BackProp(y, Nodes, Edges, Jacobians, Parameters)::IdDict
    TopoSortNodes = KahnTopoSort(Nodes, Edges)
    ChainedJacobians = IdDict{Any,Any}(y.ID => 1.0)
    for source in reverse(TopoSortNodes[1:end-1])
        CJ = false
        sinks = get(Edges, source, false)
        for sink in sinks
           
            J = ( get(ChainedJacobians, sink, false) 
                *
                get(Jacobians, (source, sink), false) )
            if (CJ == false) CJ = J
            else CJ += J end
        end
        merge!(ChainedJacobians, IdDict(source => CJ))
    end

    Gradients = IdDict()
    for source in keys(ChainedJacobians) 
        if source in keys(Parameters)
            g = get(ChainedJacobians, source, false)
            if isa(g, AbstractArray)
                g = reshape(g, size(get(Parameters, source, false)))
                # TODO: transpose?
            end
            merge!(Gradients, IdDict(source => g))
        end

    end
    return Gradients
end


function GetGradient(f, x, Parameters)::IdDict
    y, Nodes, Edges, Jacobians = ForwProp(f, x, Set(keys(Parameters)))
    return BackProp(y, Nodes, Edges, Jacobians, Parameters)
end


