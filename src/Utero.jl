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
import Base.sum, Base.adjoint

import Base.convert
import Base.promote_rule

include("Utils/ArrayUtils.jl")

include("Functionnal/Activations.jl")
export ReLU, Sigmoid
include("Functionnal/Distance.jl")
export MeanSquaredError

include("AutoDiff/ReverseOverloadingUtils.jl")
include("AutoDiff/ReverseDualReal.jl")
include("AutoDiff/ReverseDualTens.jl")
include("AutoDiff/ReverseModeAD.jl")
export ⬅Context, AddParams!, ForwardBackward!


include("Training/Step.jl")
export GradientStep!


end