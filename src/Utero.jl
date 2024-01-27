module Utero

using Random
using LinearAlgebra
using SparseArrays

import Base.:+
import Base.:-
import Base.:*
import Base.:/
import Base.:^
import Base.sin
import Base.cos
import Base.map
import Base.prod
import Base.sum

import Base.convert
import Base.promote_rule

include("Utils/ArrayUtils.jl")

include("Functionnal/Activations.jl")
export ReLU

include("AutoDiff/ReverseOverloadingUtils.jl")
include("AutoDiff/ReverseDualReal.jl")
include("AutoDiff/ReverseDualTens.jl")
include("AutoDiff/ReverseModeAD.jl")

"""
include("AutoDiffTracker/DirectedAcyclicGraph.jl")
include("AutoDiffTracker/Tracking.jl")
include("AutoDiffTracker/Jacobians.jl")
include("AutoDiffTracker/Propagation.jl")

include("Training/Loss.jl")
"""

export ⬅Context, Params, ForwardBackward

end