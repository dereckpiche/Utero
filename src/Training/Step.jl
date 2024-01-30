function GradientStep!(Δ, Params, Grads)
    for (p, g) in zip(Params, reverse(Grads))
        p.val = p.val .- Δ .* g
    end
end