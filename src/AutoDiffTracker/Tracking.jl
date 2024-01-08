
function AddNewID(IDs::Set)
    NewID = max(IDs...) + 1
    push!(IDs, NewID)
    return NewID
end

function AddParam(Parameters, val)
    NewID = 1
    if !isempty(Parameters) 
        NewID = AddNewID(Set(keys(Parameters)))
    end
    merge!(Parameters, IdDict(NewID => val))
    return NewID
end

mutable struct Tracked{T} <: Real 
    val::T
    ID::Int64 # identification in the computationnal graph
    function Tracked(val, Parameters::IdDict)
        return new{typeof(val)}(val, AddParam(Parameters, val))
    end
    Tracked(val, ID) = return new{typeof(val)}(val, ID) 
end
