module Utero

# External
using Random
using LinearAlgebra


# Internal

include("Differentiation/DirectedAcyclicGraph.jl")
include("Differentiation/Tracking.jl")
include("Differentiation/Jacobians.jl")
include("Differentiation/Propagation.jl")

end