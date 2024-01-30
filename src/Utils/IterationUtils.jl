
DataIterator(X, Y::AbstractVector) = zip(eachslice(X, dims=3), Y)
#DataIterator(X, Y) = zip(eachrow(X), eachrow(Y))

#DataIterator(Data) = zip(eachrow(Data[:, 1:-1]), eachrow(Data[:, -1:end]))