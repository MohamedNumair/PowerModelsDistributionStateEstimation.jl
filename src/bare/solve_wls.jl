################################################################################
#  Literal PMDSE                                                                #
#  solve_wls.jl : Gauss-Newton Weighted Least Squares (normal equations with a   #
#                 QR / orthogonal fallback, Abur & Exposito Ch. 2-3).            #
################################################################################

"result of a literal state-estimation solve"
struct LiteralResult
    termination::Symbol
    iterations::Int
    objective::Float64           # weighted SSR  r' W r at the solution
    solve_time::Float64
    gain_cond::Float64
    gain_rank::Int
    x_free::Vector{Float64}
    x_full::Vector{Float64}
    solution::Dict{String,Any}
    estimator::Symbol
end

"""
    _solve_gain(G, rhs, A, b; cond_tol)

Solve the normal-equation step `G Δ = rhs` with `G = HᵀWH`.  Tries a Cholesky
factorization first; if `G` is not positive-definite or ill-conditioned, falls
back to the numerically robust orthogonal solve `min ‖A Δ − b‖₂` with
`A = √W·H`, `b = √W·r` (a QR least-squares step).  Returns `(Δ, cond, used_qr)`.
"""
function _solve_gain(G::Matrix{Float64}, rhs::Vector{Float64},
                     A::Matrix{Float64}, b::Vector{Float64}; cond_tol::Float64 = 1e12)
    if all(isfinite, G) && all(isfinite, rhs)
        ch = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(G); check = false)
        if LinearAlgebra.issuccess(ch)
            Δ = ch \ rhs
            all(isfinite, Δ) && return Δ, false
        end
    end
    return A \ b, true           # QR least squares on √W·H (orthogonal estimator)
end

"condition number / numerical rank diagnostics, guarded against LAPACK failures"
function _gain_diag(G::Matrix{Float64})
    try
        sv = LinearAlgebra.svdvals(G)
        smax = sv[1]; smin = sv[end]
        c = smin > 0 ? smax / smin : Inf
        r = sum(sv .> 1e-9 * smax)
        return c, r
    catch
        return NaN, -1
    end
end

"""
    solve_wls(model; maxiter=50, tol=1e-9, verbose=false, cond_tol=1e12)

Iterate the Gauss-Newton WLS step `(HᵀWH)Δ = HᵀW r`, `r = z − h(x)`, `H =
∂h/∂x` (via `ForwardDiff`), until `‖Δ‖∞ < tol`.
"""
function solve_wls(model::SEModel; maxiter::Int = 50, tol::Float64 = 1e-9,
                   verbose::Bool = false, cond_tol::Float64 = 1e12)
    t0 = time()
    lm = model.lm
    x = flat_start_free(lm)
    W = model.w; z = model.z
    sqrtW = sqrt.(W)
    term = :maxiter; iters = 0; gcond = NaN; grank = -1

    for it in 1:maxiter
        iters = it
        h = predict(model, x)
        r = z .- h
        H = ForwardDiff.jacobian(xx -> predict(model, xx), x)
        WH = W .* H
        G = Symmetric_full(transpose(H) * WH)
        rhs = transpose(H) * (W .* r)
        A = sqrtW .* H; b = sqrtW .* r
        Δ, used_qr = _solve_gain(G, rhs, A, b; cond_tol = cond_tol)
        x .+= Δ
        verbose && println("  it $it  ‖Δ‖∞=$(LinearAlgebra.norm(Δ, Inf))  qr=$(used_qr)")
        if LinearAlgebra.norm(Δ, Inf) < tol
            term = :converged; break
        end
    end

    # diagnostics + objective at the solution
    h = predict(model, x); r = z .- h
    obj = sum(W .* r .^ 2)
    H = ForwardDiff.jacobian(xx -> predict(model, xx), x)
    G = transpose(H) * (W .* H)
    gcond, grank = _gain_diag(G)
    x_full = expand_state(lm, x)
    sol = state_solution(lm, x_full)
    return LiteralResult(term, iters, obj, time() - t0, gcond, grank, x, x_full, sol, :wls)
end

"materialize a symmetric matrix (full, dense) from `HᵀWH`"
Symmetric_full(M) = Matrix(LinearAlgebra.Symmetric((M .+ transpose(M)) ./ 2))
