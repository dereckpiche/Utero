This is the plan for implementing reverse-mode automatic differentiation in a very minimal and flexible way using good old c-flavoured c++. All the fancy stuff will be built on fundamental operations that are overloaded. These functions record a pointer to the right backward function. Say we have $c(x) = f^{n} \circ f^{n-1} \circ \dots f^1 (x)$. Then $f_{back}^{i}(x) := \nabla_{f_{back}^{i}} f_{back}^{i-1} \cdot x$ such that $f_{back}^{i}(\nabla*{c} f_{back}^{i-1}) = \nabla_{c} f_{back}^{i}$. The basic operations that will be overloaded are

- matrix multiplication
- element-wise operations between two tensors (summation, substraction, division, multiplication, exponentiation, etc)
- broadcasted functions (relu, sigmoid, sin, cos, normalization, etc)
- tensor reshape operations (concatenation, transpose, reshape)

The library uses a special struct called "tensor" like pytorch. All of the manipulations in the forward pass of a neural net must act on tensors, no other types are allowed. Some values are constants that should not be tracked. Operations must be saved in order to compute the backward pass. Every forward pass can be decomposed as a bunch of fundamental operations taking a bunch of parent tensors (often two) and producing a new child tensor. Per the chain rule, if the jacobians of the child with respect to the parents and the gradient of the cost with respect to the child are matrix multiplied, the result will be the gradients of the cost with respect to the parents. Therefore, when a fundamental operation $f$ (tracked) produces a child, the child is given an $f_{back}$ function ($f_{back}(P_1, P_2, \Delta C) = \Delta P_1, \Delta P_2$). The $f_{back}$ takes the values of the parents and the gradient of the child and returns the gradients of the parents. The parents then use their own $f_back$'s and send the results to their own parents. Thus so,backpropagation eventually reaches the first tensors, which have no parents. The gradient structure should be have
- a pointer to the value tensor
- a pointer to the gradient tensor
- a pointer to the $f_{back}$ function
- 

A simple queue can be implemented. Using the above structure gives an implicit tape with topological order. When the queue has been emptied, all of the gradient will have been computed. At a higher level, there will be an array which contains the pointers to all of the parameters of the network. This array will be iterated such that all parameter tensors are accessed and their value arrays updated with a gradient step. 

# pseudocode
The overloaded functions come in likeso pairs:
```cpp
tensor foo(tensor x, tensor y){
    tensor z = // (...) 
    tape.append(&z); // tape is a global doubly-linked chain
    z.fback = &fback_pointer; // pointer to backpropagation function
    return z;
}

tensor* foo_back(tensor grad_child, tensor parent_x, tensor parent_y){
    tensor grad_par_x = // (...);
    tensor grad_par_y = // (...);
    tensor* parent_grads = // (...);
    return parent_grads;
}
```

Then we have 

    