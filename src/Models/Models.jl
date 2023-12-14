


mutable struct Sequential <: Model
    network
    parameters::IdDict #(TrackedIds => Tracked)
end


mutable struct Reccurent <: Model
    network
    parameters::IdDict
end

