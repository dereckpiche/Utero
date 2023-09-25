import Base.+
abstract type Mutator end


""" 
Every tensor is a parameter. 
We can reduce our model into atomic operations of the form
x::Vector -> f(T) -> y::Vector
for every element of the tensor T, we can create a jacobian matrix
where J[i, j] is d y_i / d f(E, y_i).
However, we can save much space for some operations, since
linear operations link rows only to the corresponding row 
of the y vector. 



Jacobian is 
J[i,j] = dv_i / dy_i ??? <- verify
"""
function GetJacoDict(m::Mutator)
    dict = IdDict()
    for p in propertynames(m) 
        push!(dict, getfield(m, p)=>nothing)
    end
    return dict
end

function GetJacoDict(d::IdDict, m::Mutator)
    return merge(d, GetJacoDictI(m))
end


"""
Computes the values at every layer and 
creates a computationnal graph description in a dictionnary
which will allow us to go backwards.
# The first input x is turned into a Tag
# everything it touches becomes a Tag
# thus, operations will come in op(Tag, Tensor)
# in this comb, we compute the and 

# Tag:
# for op(T, x::Tag)->y, (T, y) is the Tag
# contains (parameter origin object, y)

# for each basic op(T, x::Tag),
# add (t => x[1]) to CompGraph
# and return (T, y), which is the Tag

# based on the "y" object id, we can go backwards in the computationnal graph
# and compute the gradients

# Fix: how to know which one we are deriving. Check if object already 
in Jacobians dictionnary

""" 

struct Param{T}
    v::T
    id::Float64
end


Base.:+(x::Param, y::Param) = 1
x = 3
y = 4
x = Param(x)
y = Param(y)

print(x+y)
#@eval Base.:+(x::Int, y::Int) = 1

function Forward(m::Function, x)
    Jacobians = IdDict()
    CompGraph = IdDict()

    """
    function +(x::AbstractArray, t::AbstractArray)
        y = Base.+(x, t)
        push!(Jacobians, (t => t))
        push!(CompGraph, (t => (+, x)))
        return Tag{typeof(t), typeof(y)}(t, y)
    end
    """

    d = 3+3
    println(d)

    
    #y = m(x)


    #return y, CompGraph, Jacobians, 4+4
    return 

    #return y, CompGraph, Jacobians
    
end




"""
We use reverse mode-autodifferentiation with the computationnal graph
Backprop. Compute the gradients by going backwards using
the comp graph computed by 'Forward'. Backprop will skip 
computing the gradients for excluded methods!
"""

function Backprop(m, y, CompGraph, Jacobians)
    function gradient(+, f::Array, t::Array)
        # use chain rule specifically for + operation
        return
    end
    return
end




"""
Here is how to proceed
We create a new type, Param. 
Each param has a tensor and an identifier.
We also create a new second new type, 
TagParam. This type contains 
- the identifier
- the tensor
- the tensor's gradient value at x 
It would be nice if we didn't have to store
the information about the identifier an the gradient 
directly in a struct, maybe use macros.
We would want only the resulting function to return an output 
and a dict. This dict is important. See how pytorch saves the 
gradient for every tensor? Do they give codenames? Do we ask the user for 
code names? Probably best that we generate them ourselves for every value.
So, the user can specify a layer name if he choses. If not, our 
constructors will generate a random one. This will be great for debugging! 

We can simply get the name of a variable. The gradient should look like 
{ dense1 : {w:grad, b:grad, f:nothing}, attention2 : {w:grad, b:grad, f:nothing} }
No part of the network can have the same name! Will cause an error!
The code should work with simple julia function like x**2, so lets make it so that 
our function is defined for basic non-structure as well as structures (composites)
of the type Mutator. 

Will need to define gradient 
for matrices and vectors (addition (ones), matmul, hadamard product, hadamard division)


"""

"""
macro rules()
    #eval(:(x::Number + y::Number = x*y))
    return :(x::Number + y::Number = x*y)
end

function basicRules()
    x + y = x*y
    x * y = x-y
end

rules = quote
    f(y) = x
    # single values
    g::Tag + f::Tag = (g[1]+f[1], g[2]+f[2])  
    g::Tag * f::Tag = (g[1]*f[1], f[1]*g[2] + f[2]*g[1])
    (g::Mutator)(f::Tag) = (g(f[1])[1], g(f(1))[2]*f[2]) # something like that

    
    # chain rule 
    #function (g::Mutator ∘ f::Mutator)(x::Array)
    #    fTag = f(x)
    #end


end
"""