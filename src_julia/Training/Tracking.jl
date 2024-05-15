
"""
Track(_, _): add hook to global tracking dict.
Track(d::Dense): implicitely add to global tracking dictionnary.
"""

function Track(Nodes, Tensor)
    Tensor = Tracked(Tensor, Nodes)
end


function Track(d::Dense)
    return
end

function Track(Nodes, d::Dense)
    Track(d.W)
    Track(d.B)
    d.B = Tracked(B, Vector)
end

"""
Untrack
"""
function Untrack(m::Model)
    for t in values(m.Tracked)
        t = t.val
    end
    m.Tracked = IdDict()
    return
end




