module Utero
using Random
using LinearAlgebra
using SparseArrays
using UUIDs

import Base.:+
import Base.:-
import Base.:*
import Base.:/
import Base.:^
import Base.sin
import Base.cos
import Base.map
import Base.prod
import Base.broadcast, Base.broadcasted, Base.dropdims
import Base.sum, Base.adjoint, Base.getindex

import Base.convert
import Base.promote_rule

include("Utils/ArrayUtils.jl")
export OneHot
include("Utils/IterationUtils.jl")
export DataIterator

include("Functionnal/Activation.jl")
export ReLU, Sigmoid
include("Functionnal/Distance.jl")
export MeanSquaredError
include("Functionnal/Normalization.jl")
export Normalize, Softmax

include("AutoDiff/ReverseOverloading.jl")
include("AutoDiff/ReverseDual.jl")
export ⬅Dual
include("AutoDiff/ReverseModeAD.jl")
export ⬅Context, ⬅CleanContext!, AddParams!, ForwardBackward!

include("Nets/Dense.jl")
export Dense
include("Nets/Sequential.jl")
export Sequential

include("Nets/Sequential.jl")

include("Training/Step.jl")
export GradientStep!


end