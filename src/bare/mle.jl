################################################################################
#  Literal PMDSE                                                                #
#  mle.jl : general Maximum-Likelihood interface for the literal estimator.      #
#                                                                              #
#  The WLS estimator is the special case of MLE under independent Gaussian       #
#  measurement errors.  This file exposes the likelihood abstraction so that an  #
#  arbitrary per-measurement distribution (any `Distributions` /                 #
#  `Polynomials`/`ExtendedBeta`, mirroring the `mle` branch of                   #
#  `constraint_mc_residual`) can be plugged in with **no change** to the         #
#  measurement model `h(x)` or the references.                                   #
#                                                                              #
#  Objective:  maximize  ℓ(x) = Σ_i logpdf(dst_i, h_i(x))                        #
#  solved by a (Levenberg-damped) Newton iteration with `ForwardDiff` gradient   #
#  and Hessian.  For `dst_i = Normal(z_i, σ_i)` this is exactly WLS.             #
################################################################################

"default per-row measurement distributions implied by the WLS weights (Gaussian)"
function gaussian_row_dsts(model::SEModel)
    return [_DST.Normal(model.z[i], 1.0 / sqrt(model.w[i])) for i in eachindex(model.z)]
end

"log-likelihood of one row given the model prediction `ĥ` (Gaussian / general)"
loglik_row(dst::_DST.Distribution, ĥ) = _DST.logpdf(dst, ĥ)
loglik_row(dst::Real, ĥ) = -((ĥ - dst)^2)          # degenerate hard value

"""
    solve_mle(model; row_dst=nothing, maxiter=50, tol=1e-9, verbose=false)

Maximum-likelihood state estimate by **Fisher scoring**: each iteration solves
`(Hᵀ Λ H) Δ = Hᵀ g`, where `g_i = ∂logpdf(dst_i, h_i)/∂h_i` is the per-row score
and `Λ_i = −∂²logpdf/∂h_i²` the per-row information (both via `ForwardDiff`).  The
step reuses the WLS QR / orthogonal fallback, so it is robust on ill-conditioned
gains.  For `dst_i = Normal(z_i, σ_i)` one has `g_i = (z_i−h_i)/σ_i²`,
`Λ_i = 1/σ_i²`, so the step is exactly the WLS normal-equation step — `solve_mle`
then returns the WLS estimate (verified in the tests).  When `row_dst === nothing`
the Gaussian distributions implied by the WLS weights are used; pass `row_dst`
(aligned with the residual rows) for a general likelihood.
"""
function solve_mle(model::SEModel; row_dst = nothing, maxiter::Int = 50,
                   tol::Float64 = 1e-9, verbose::Bool = false)
    t0 = time()
    lm = model.lm
    dsts = row_dst === nothing ? gaussian_row_dsts(model) : row_dst
    score1(i, η) = ForwardDiff.derivative(t -> loglik_row(dsts[i], t), η)
    info1(i, η)  = -ForwardDiff.derivative(t -> score1(i, t), η)

    x = flat_start_free(lm); term = :maxiter; iters = 0
    for it in 1:maxiter
        iters = it
        h = predict(model, x)
        H = ForwardDiff.jacobian(xx -> predict(model, xx), x)
        g = [score1(i, h[i]) for i in eachindex(h)]               # score
        Λ = [max(info1(i, h[i]), 1e-12) for i in eachindex(h)]    # information (PSD)
        F = Symmetric_full(transpose(H) * (Λ .* H))
        score = transpose(H) * g
        A = sqrt.(Λ) .* H; b = g ./ sqrt.(Λ)                      # A'A=F, A'b=score
        Δ, _ = _solve_gain(F, score, A, b)
        x .+= Δ
        verbose && println("  mle it $it  ‖Δ‖∞=$(LinearAlgebra.norm(Δ, Inf))")
        LinearAlgebra.norm(Δ, Inf) < tol && (term = :converged; break)
    end

    obj = -sum(loglik_row(dsts[i], hi) for (i, hi) in enumerate(predict(model, x)))
    H = ForwardDiff.jacobian(xx -> predict(model, xx), x)
    gcond, grank = _gain_diag(transpose(H) * (model.w .* H))
    x_full = expand_state(lm, x)
    return LiteralResult(term, iters, obj, time() - t0, gcond, grank, x, x_full,
                         state_solution(lm, x_full), :mle)
end
