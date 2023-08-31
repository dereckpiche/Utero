import Base.+
abstract type Mutator end


""" 
Every tensor is a parameter. 
We can reduce our model into atomic operations of the form
x::Vector -> f(T) -> y::Vector
for every element of the tensor T, we can create a jacobian matrix
where J[i, j] is d y_i / d f(E, y_i).



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
""" 
function Forward(m::Function, x)
    CompGraph = IdDict()
    Jacobians = IdDict()

    x -> Dual # x becomes a Dual
    # everything it touches becomes a Dual
    # thus, operations will come in op(Dual, Tensor)
    # in this comb, we compute the and 

    function +(x::Dual, t::Array)
        # Add x => t to CompGraph
        # 
    end

    function *(x::Dual, t::Array)
        
    end


    y = m(x)

    # based on the "y" object id, we can go backwards in the computationnal graph
    # and compute the gradients

    return y, CompGraph, Jacobians
    
end


"""
Backprop. Compute the gradients by going backwards using
the comp graph computed by 'Forward'. Backprop will skip 
computing the gradients for excluded methods!
"""

function Backprop(m, y, CompGraph, Jacobians)
    return
end

function Jacobian(cost::Mutator, x::AbstractArray)
    jaco = 2 # params dict
    #g::Dual + f::Dual = (g[1]+f[1], g[2]+f[2])  
    #g::Dual * f::Dual = (g[1]*f[1], f[1]*g[2] + f[2]*g[1])
    #(g::Mutator)(f::Dual) = (g(f[1])[1], g(f(1))[2]*f[2]) # something like that
end



"""
Here is how to proceed
We create a new type, Param. 
Each param has a tensor and an identifier.
We also create a new second new type, 
DualParam. This type contains 
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
    g::Dual + f::Dual = (g[1]+f[1], g[2]+f[2])  
    g::Dual * f::Dual = (g[1]*f[1], f[1]*g[2] + f[2]*g[1])
    (g::Mutator)(f::Dual) = (g(f[1])[1], g(f(1))[2]*f[2]) # something like that

    
    # chain rule 
    #function (g::Mutator ∘ f::Mutator)(x::Array)
    #    fDual = f(x)
    #end


end
"""