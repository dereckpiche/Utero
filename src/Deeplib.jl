module Deeplib

# External
using Zygote # automatic differentiation
using Random


# Internal
include("dense.jl")
include("activations.jl")
include("train.jl")
include("optimizers.jl")
include("loss.jl")
end