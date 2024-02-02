using MLDatasets
include("../src/Utero.jl")
using .Utero 

TrainData = MNIST(:train)
ValidData = MNIST(:test)

ctx = ⬅Context()

densenet = Sequential(Dense(28*28, 16, ReLU), Dense(16, 16, ReLU), Dense(16, 10, ReLU), Softmax)
display(densenet(rand(28*28, 1)))
display(MeanSquaredError(densenet(rand(28*28, 1)), rand(10, 1)))
AddParams!(ctx, densenet)
println(typeof(densenet.Sequence[1].W))
for (x, y) in DataIterator(TrainData.features, TrainData.targets)
    x = reshape(x, 28*28, 1)
    y = OneHot(Int64(y+1), 10)
    display(densenet.Sequence[1].W)
    (out, paramgrads) = ForwardBackward!(ctx, x -> MeanSquaredError(densenet(x), y), x)
    GradientStep!(0.0001, ctx.Params, paramgrads)
    display(paramgrads[1])
    display(densenet.Sequence[1].W)
    exit("")
end



