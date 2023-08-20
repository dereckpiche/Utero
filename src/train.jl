

function train(
    x, 
    y, 
    model, 
    #optimiser::Function, 
    lossF,
    #batchSize::Int,
    epochs,
    stepSize)
    iterations = size(x)[1]
    for _ in 1:epochs
        for i in 1:iterations
            println(lossF(model(x[i, :]), y[i, :]))
            model = descent(x[i, :], y[i, :], lossF, model, stepSize)
        end
    end
end