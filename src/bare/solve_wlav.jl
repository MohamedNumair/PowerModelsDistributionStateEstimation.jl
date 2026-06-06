################################################################################
#  Literal PMDSE                                                                #
#  solve_wlav.jl : Weighted Least Absolute Value via Iteratively Reweighted     #
#                  Least Squares (IRLS).  Minimises Σ (1/(rescaler·σ_i)) |r_i|,  #
#                  matching the `wlav` criterion of `constraint_mc_residual`.    #
################################################################################

"""
    solve_wlav(model; maxiter=60, tol=1e-9, huber=1e-4, verbose=false)

IRLS solution of the WLAV problem.  Each outer iteration solves a Gauss-Newton
WLS step with weights `w̃_i = √w_i / max(|r_i|, huber)` so that the quadratic
surrogate `Σ w̃_i r_i²` upper-bounds the WLAV objective `Σ √w_i |r_i|` (Huber
relaxation near `r=0`).  WLAV is robust to single gross errors, unlike WLS.
"""
function solve_wlav(model::SEModel; maxiter::Int = 60, tol::Float64 = 1e-9,
                    huber::Float64 = 1e-4, verbose::Bool = false)
    t0 = time()
    lm = model.lm
    x = flat_start_free(lm)
    wbase = model.w; z = model.z
    sw = sqrt.(wbase)                          # √w_i  (= 1/(rescaler σ_i))
    term = :maxiter; iters = 0

    for it in 1:maxiter
        iters = it
        h = predict(model, x); r = z .- h
        w̃ = sw ./ max.(abs.(r), huber)         # IRLS weights
        H = ForwardDiff.jacobian(xx -> predict(model, xx), x)
        WH = w̃ .* H
        G = Symmetric_full(transpose(H) * WH)
        rhs = transpose(H) * (w̃ .* r)
        A = sqrt.(w̃) .* H; b = sqrt.(w̃) .* r
        Δ, _ = _solve_gain(G, rhs, A, b)
        x .+= Δ
        verbose && println("  wlav it $it  ‖Δ‖∞=$(LinearAlgebra.norm(Δ, Inf))")
        LinearAlgebra.norm(Δ, Inf) < tol && (term = :converged; break)
    end

    h = predict(model, x); r = z .- h
    obj = sum(sw .* abs.(r))                   # WLAV objective Σ √w_i |r_i|
    H = ForwardDiff.jacobian(xx -> predict(model, xx), x)
    gcond, grank = _gain_diag(transpose(H) * (wbase .* H))
    x_full = expand_state(lm, x)
    sol = state_solution(lm, x_full)
    return LiteralResult(term, iters, obj, time() - t0, gcond, grank, x, x_full, sol, :wlav)
end
