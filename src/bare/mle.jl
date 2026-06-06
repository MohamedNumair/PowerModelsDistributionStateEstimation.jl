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
    solve_mle(model; row_dst=nothing, maxiter=50, tol=1e-9, λ0=1e-6, verbose=false)

Maximum-likelihood state estimate.  When `row_dst === nothing` the Gaussian
distributions implied by the WLS weights are used and the result is identical to
`solve_wls` (validated in the tests).  Provide `row_dst` (a vector of
distributions aligned with the residual rows) to use the general likelihood.
"""
function solve_mle(model::SEModel; row_dst = nothing, maxiter::Int = 50,
                   tol::Float64 = 1e-9, λ0::Float64 = 1e-6, verbose::Bool = false)
    t0 = time()
    lm = model.lm
    dsts = row_dst === nothing ? gaussian_row_dsts(model) : row_dst
    negℓ(x) = -sum(loglik_row(dsts[i], hi) for (i, hi) in enumerate(predict(model, x)))

    x = flat_start_free(lm); term = :maxiter; iters = 0
    for it in 1:maxiter
        iters = it
        g = ForwardDiff.gradient(negℓ, x)
        Hn = ForwardDiff.hessian(negℓ, x)
        λ = λ0
        Δ = nothing
        for _ in 1:30                                   # Levenberg damping until SPD solvable
            M = Hn + λ * LinearAlgebra.I
            ch = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(M); check = false)
            if LinearAlgebra.issuccess(ch)
                Δcand = ch \ g
                if all(isfinite, Δcand); Δ = Δcand; break; end
            end
            λ *= 10
        end
        Δ === nothing && (Δ = LinearAlgebra.pinv(Hn) * g)
        x .-= Δ
        verbose && println("  mle it $it  ‖Δ‖∞=$(LinearAlgebra.norm(Δ, Inf))")
        LinearAlgebra.norm(Δ, Inf) < tol && (term = :converged; break)
    end

    obj = negℓ(x)
    H = ForwardDiff.jacobian(xx -> predict(model, xx), x)
    gcond, grank = _gain_diag(transpose(H) * (model.w .* H))
    x_full = expand_state(lm, x)
    return LiteralResult(term, iters, obj, time() - t0, gcond, grank, x, x_full,
                         state_solution(lm, x_full), :mle)
end
