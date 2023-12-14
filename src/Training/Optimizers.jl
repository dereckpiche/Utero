"""
To update models with respect to loss function
"""


function DescentStep(x, y, lossF, model, stepSize)
    grad = Zygote.gradient(m -> lossF(m(x), y), model)[1]
    for key in keys(grad)
        param = getproperty(model, key)
        if (grad[key] != nothing)
            delta = grad[key] * -stepSize
            setproperty!(model, key, param + delta)
        end
    end
    return model
end


function BatchDescentStep(xB, yB, lossF, model, stepSize)
    return
end