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

## Tensorial Jacobian of Matrix Multiplication
Let $f(X) = X \times Y$. Then $f: \mathbb{R}^{M_x \times N_x} \to \mathbb{R}^{M_x \times N_Y}$. 
We have $$f(X)_{i, j} = \sum_k X_{i, k} Y_{k, j}$$ Thus,
$$\frac{\partial f(X)_{i,j}}{\partial X_{s, t}} = 
\begin{cases}
Y_{j, t} & \text{if $s = i$} \\
0 & \text{else}
\end{cases}$$
Let $f(Y) = X \times Y$. Then $f: \mathbb{R}^{M_y \times N_y} \to \mathbb{R}^{M_x \times N_Y}$
Thus, 
$$\frac{\partial f(Y)_{i,j}}{\partial Y_{s, t}} = 
\begin{cases}
X_{i, s} & \text{if $t = j$} \\
0 & \text{else}
\end{cases}$$


