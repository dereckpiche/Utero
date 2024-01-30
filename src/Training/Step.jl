function GradientStep!(Δ, Params, Grads)
    for (p, g) in zip(Params, Grads)
        p.val -= Δ * g
    end
end