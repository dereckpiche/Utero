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
    for f in S.Sequence 
        f = AddParams!(ctx, f) 
    end
end
