
function Accuracy(probabilities::AbstractArray, labels::AbstractVector)
    accurate = 0
    predicted = [argmax(row) for row in eachrow(probabilities)]
    c = 0
    for (pred, label) in zip(predicted, labels)
        if c < 30
            @show pred
            @show label
        end
        c += 1
        accurate += Int64(pred == label)
    end
    return accurate / length(labels)
end