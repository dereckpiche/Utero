module Deeplib

# External
using Zygote # automatic differentiation
using Random


# Internal
include("AutoDiff.jl")
include("Dense.jl")
include("Activations.jl")
include("Train.jl")
include("Optimizers.jl")
include("Loss.jl")
end