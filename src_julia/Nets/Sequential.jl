mutable struct Sequential
    Sequence #::Array{Any, Functions}
    function Sequential(models...)
        return new([m for m in models])
    end
end

function (S::Sequential)(X)
    for F in S.Sequence X = F(X) end
    return X
end

function AddParams!(ctx::⬅Context, S::Sequential)
    for layer in S.Sequence 
        isa(layer, Mutator) ? layer = AddParams!(ctx, layer) : nothing
    end
end

function Untrack!(S::Sequential)
    for layer in S.Sequence
        isa(layer, Mutator) ? Untrack!(layer) : nothing
    end
end