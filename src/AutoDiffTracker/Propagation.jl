
TwoTensorOperations = (
    :+, :-, :*, :/, :^
)

PairedTypes = (
    :Integer, :AbstractFloat, :Array
)

SingleTensorOperations = (
    :map, :sum, :prod, :cos, :sin
) 


function AddJacobian(Jacobians, source, sink, Jacobian)
    # adds jacobian of sink with respect to source to jacobians
    merge!(Jacobians, IdDict((source, sink) => Jacobian)) 
end


function ForwProp(f, w, Parameters::IdDict)

    # Overcharge the operators to create a computationnal graph as well as 
    # the intermediate jacobians for backpropagation at a later stage

    global NodeIds = deepcopy(Set(keys(Parameters)))
    global Edges = IdDict()
    global Jacobians = IdDict()

    for t in PairedTypes for op in TwoTensorOperations
        @eval begin
            global function ($op)(a::T, b::Tracked) where {T<:($t)}
                ID = AddNewID(NodeIds)
                J = GetJacobian(($op), a, b)
                AddEdge(Edges, b.ID, ID)
                AddJacobian(Jacobians, b.ID, ID, J)
                return Tracked(($op)(a, b.val), ID)
            end

            global function ($op)(a::Tracked, b::T) where {T<:($t)}
                ID = AddNewID(NodeIds)
                J = GetJacobian(($op), a, b)
                AddEdge(Edges, a.ID, ID)
                AddJacobian(Jacobians, a.ID, ID, J)
                return Tracked(($op)(a.val, b), ID)
            end

            global function ($op)(a::Tracked, b::Tracked)
                ID = AddNewID(NodeIds)
                Ja = GetJacobian(($op), a, b.val)
                AddEdge(Edges, a.ID, ID)
                AddJacobian(Jacobians, a.ID, ID, Ja)
                Jb = GetJacobian(($op), a.val, b)
                AddEdge(Edges, b.ID, ID)
                AddJacobian(Jacobians, b.ID, ID, Jb)
                return Tracked(($op)(a.val, b.val), ID)
            end
        end
    end end


    for op in SingleTensorOperations
        @eval begin
            global function ($op)(a::Tracked)
                ID = AddNewID(NodeIds)
                J = GetJacobian(($op), a)
                AddEdge(Edges, a.ID, ID)
                AddJacobian(Jacobians, a.ID, ID, J)
                return Tracked(($op)(a.val), ID)
            end

            global function ($op)(a::Tracked, args...)
                ID = AddNewID(NodeIds)
                J = GetJacobian(($op), a, args...)
                AddEdge(Edges, a.ID, ID)
                AddJacobian(Jacobians, a.ID, ID, J)
                return Tracked(($op)(a.val, args...), ID)
            end
        end
    end

    y = Base.invokelatest(f, w)
    return (y, NodeIds, Edges, Jacobians, Parameters)

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


function GetGradient(f::Any, w::Tracked, Parameters)::IdDict
    return BackProp(ForwProp(f, w, Parameters)...)
end

function GetGradient(f::Any, nothing, Parameters)::IdDict
    return BackProp(ForwProp(f, nothing, Parameters)...)
end

function GetGradient(f::Any, x::AbstractFloat)::IdDict
    params = IdDict()
    x = Tracked(x, params)
    g = BackProp(ForwProp(f, x, params)...)
    return values(g)[1]
end

function GetGradient(f::Any, Parameters::IdDict) # implicit
    return GetGradient(f, nothing, Parameters)
end