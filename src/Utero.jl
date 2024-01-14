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

include("AutoDiffByDual/ReverseDual.jl")
include("AutoDiffByDual/ReverseOL.jl")
include("AutoDiffByDual/ReverseModeAD.jl")

"""
include("AutoDiffTracker/DirectedAcyclicGraph.jl")
include("AutoDiffTracker/Tracking.jl")
include("AutoDiffTracker/Jacobians.jl")
include("AutoDiffTracker/Propagation.jl")

include("Training/Loss.jl")
"""

export ⬅Context, Params, ForwardBackward

end