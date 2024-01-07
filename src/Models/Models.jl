


mutable struct Sequential <: Model
    F
    Parameters::IdDict #(Ids => Tracked) retains which tensor could be tracked
    Tracked::IdDict #(TrackedIds => Tracked) contains the currently tracked tensors
end


mutable struct Reccurent <: Model
    F
    Parameters::IdDict #(Ids => Tracked) retains which tensor could be tracked
    Tracked::IdDict #(TrackedIds => Tracked) contains the currently tracked tensors
end

