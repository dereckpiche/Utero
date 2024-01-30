#  Reverse Duals
Let 
$$c(x_1, x_2, \dots, x_n) = (x_1, x_2, \dots) \to (z_1, z_2, ,\dots) \to y \to c$$
we would like to obtain $\frac{\partial c}{\partial x_i}$. Then, simply a simple 

During the forward pass, each output is accompagnied by a function containing the derivatives used to compute it. This function is called the Chainer.
Then, knowing $\frac{dc}{dy}$, we use the $y$'s Chainer to obtain the partial derivatives of the $z$'s. 
$$(\frac{\partial c}{\partial z_i}, \frac{\partial c}{\partial z_i}, \dots) = L_{z}(\frac{dc}{dy})$$ 

Then, we can use the Chainers of each $z_i$ to obtain the partial derivative of the cost with respect to any $x_i$:
$$\frac{\partial c}{\partial x_k} = \sum_i \left[L_{x}(\frac{\partial c}{\partial z_i}) \right]_k$$


# The Tensorial Jacobian
Let a $f : \mathbb{R}^{C_1, C_2, \dots} \to \mathbb{R}^{K_1, K_2, \dots}$. The tensorial Jacobian for this mapping is the tensor $J$ which respects
$$
J_{k_1, k_2, \dots, c_1, c_2, \dots} = \frac{\partial f(X)_{k_1, k_2, \dots}}{\partial X_{c_1, c_2, \dots}}
$$

## Chain Rule for the Tensorial Jacobian
Let $f : \mathbb{R}^{C_1, C_2, \dots} \to \mathbb{R}^{K_1, K_2, \dots}$,
$g : \mathbb{R}^{V_1, V_2, \dots} \to \mathbb{R}^{C_1, C_2, \dots}$
and $h(x) = f \circ g (x)$. Then $h : \mathbb{R}^{V_1, V_2, \dots} \to \mathbb{R}^{K_1, K_2, \dots}$. 

In Utero, we introduce the Tensobian
$$Ṫ_h[k_1, k_2, \dots; v_1, v_2, \dots] := \frac{\partial h_{k_1, k_2, \dots}}{\partial x_{v_1, v_2, \dots}}$$

The chain rule
$$
\begin{align}
    \frac{d h}{dx} = \frac{d f \circ g}{dx} = \frac{d f}{d g}\frac{d g}{d x}\tag{Chain Rule}
\end{align}
$$

can be generalized to tensor notation in this way:
$$
\begin{align}
    Ṫ(X \to H)^{k_1, k_2, \dots}_{v_1, v_2, \dots}
    :=
    \frac{\partial H_{k_1, k_2, \dots}}{\partial X_{v_1, v_2, \dots}}
    = 
    \sum_{c_1,c_2,\dots \in C_1 \times C_2 \times \dots} \frac{\partial F_{k_1, k_2, \dots}}{\partial G_{c_1,c_2,\dots}}\frac{\partial G_{c_1,c_2,\dots}}{\partial X_{v_1, v_2, \dots}}\tag{Tensobian Chain Rule}
\end{align}
$$
where $\sum_{}$ is an abbreviation of $$. We can also use Einstein summation notation for an implicit sum:

## Gradient of Matrix Multiplication

Let $F(X) = X \times Y$. Then $F: \mathbb{R}^{m_x \times n_x} \to \mathbb{R}^{m_x \times n_y}$
Let $G : \mathbb{R}^{m_x \times n_y} \to \mathbb{R}$.
Let $H := G \circ F$. We want to find $\nabla_H$ knowing $\nabla_G$.  
We have $$F(X)_{p, q} = \sum_k X_{p, k} Y_{k, q}$$ 
Thus,
$$
\left[ \nabla_F \right]^{p, q}_{i, j}
:= 
\frac{\partial H_{p, q}}{\partial X_{i, j}}
=
\begin{cases}
Y_{j, q} & \text{if $p = i$} \\
0 & \text{else}
\end{cases}$$

Using the chain rule, 
$$\begin{align}
    \left[ \nabla_H \right]_{i, j}
    =
    \sum_p \sum_q \left[ \nabla_G \right]_{p, q} \left[ \nabla_F \right]^{p, q}_{i, j}
    =
    \sum_q \left[ \nabla_G \right]_{i,q} Y_{j, q}
    =
    \sum_q \left[ \nabla_G \right]_{i,q} Y^\top_{q, j}
\end{align}$$
Thus,
$$ \nabla_H = \nabla_G \times Y^\top $$

___
Since
$$X \times Y \equiv (Y^\top \times X^\top)^\top$$ 
the gradient with respect to $Y$ is 
$$\begin{align}
    \nabla_H
    =
    (\nabla_G^\top \times (X^\top)^\top)^\top
    =
    X^\top \times \nabla_G
\end{align}$$








