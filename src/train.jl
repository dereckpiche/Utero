

function train(x::Array, 
    y::Array, 
    model::Function, 
    optimiser::Function, 
    lossF::Function,
    batchSize::Int,
    epochs::Int)
    for _ in 1:epochs
        for i in 1:iterations
        y = model, x
        cost = lossF, yTruthB

end