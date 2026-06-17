################################################################################
#  Literal PMDSE                                                                #
#  literal_pmdse.jl : user-facing entry point `solve_mc_se_literal`.            #
#                                                                              #
#  Solves the 4-wire IVR explicit-neutral DSSE with explicit matrices           #
#  (Gauss-Newton WLS / IRLS WLAV / general MLE) on the *same* PMD mathematical   #
#  data dictionary (`data_math` + `data_math["meas"]` + `data_math["se_settings"]`)#
#  consumed by `solve_ivr_en_mc_se` (`IVRENPowerModel`).  No JuMP is used.       #
################################################################################

"""
    solve_mc_se_literal(data; estimator=:wls, method=nothing, reference=:sota,
                        ref_values=nothing, maxiter=50, tol=1e-9, verbose=false, kwargs...)

Run the literal (matrix-based, JuMP-free) explicit-neutral state estimator.

# Arguments
- `data` : PMD **mathematical** data dictionary, with `data["meas"]` populated
  (e.g. via `add_measurements!`) and an optional `data["se_settings"]`
  (`"rescaler"`, and optionally `"reference"`).
- `estimator` : `:wls` (default), `:wlav`, or `:mle`.
- `method` : selects the WLS *solve option* when `estimator == :wls`:
  * `nothing` (default) — the in-house Gauss–Newton WLS (`solve_wls`).
  * `:iterative_linear` — the PowerGridModel default: measurements are linearised
    to complex currents/voltage-phasors each iteration and a constant linear WLS
    system is re-solved ([`solve_se_il`](@ref)).
  * `:newton_raphson` — PowerGridModel's nonlinear Gauss–Newton, warm-started
    from one `iterative_linear` solve ([`solve_se_nr`](@ref)).
  Both PGM methods converge to the same WLS optimum; they are validated against
  PowerGridModel's own state-estimation examples (see `test/literal`).
- `reference` : `:sota` (ground reference-bus neutral + one angle datum, default),
  `:full_slack` (fix the whole reference-bus phasor), or `:prop` (fix only the
  grounded/neutral terminals — the slack-angle gauge is then taken from a voltage
  phasor measurement, or pinned automatically when only magnitudes are measured).
- `ref_values` : optional `(vr_dict, vi_dict)` over reference-bus terminals for
  `:full_slack`.

Returns a [`LiteralResult`](@ref) with `solution`, `objective`, `iterations`,
`solve_time`, `gain_cond`, `gain_rank` and `termination`.
"""
function solve_mc_se_literal(data::Dict{String,<:Any}; estimator::Symbol = :wls,
                             method::Union{Nothing,Symbol} = nothing,
                             reference::Symbol = :sota, ref_values = nothing,
                             maxiter::Int = 50, tol::Float64 = 1e-9,
                             verbose::Bool = false, kwargs...)
    haskey(data, "meas") || error("solve_mc_se_literal: data has no \"meas\" dictionary")
    se = get(data, "se_settings", Dict{String,Any}())
    rescaler = Float64(get(se, "rescaler", 1.0))
    reference = Symbol(get(se, "reference", reference))
    method = method === nothing ? (haskey(se, "method") ? Symbol(se["method"]) : nothing) : method

    lm = LiteralModel(data; reference = reference, ref_values = ref_values)

    # PowerGridModel solve options (atom-based, four-wire general)
    if estimator == :wls && method in (:iterative_linear, :newton_raphson)
        atoms = build_se_atoms(lm, data; rescaler = rescaler)
        return method == :iterative_linear ?
               solve_se_il(lm, atoms; maxiter = maxiter, tol = tol, verbose = verbose) :
               solve_se_nr(lm, atoms; maxiter = maxiter, tol = tol, verbose = verbose)
    end

    model = build_se_model(lm, data; rescaler = rescaler)
    if estimator == :wls
        return solve_wls(model; maxiter = maxiter, tol = tol, verbose = verbose)
    elseif estimator == :wlav
        return solve_wlav(model; maxiter = maxiter, tol = tol, verbose = verbose)
    elseif estimator == :mle
        return solve_mle(model; maxiter = maxiter, tol = tol, verbose = verbose)
    else
        error("solve_mc_se_literal: unknown estimator :$(estimator) (use :wls, :wlav or :mle)")
    end
end

# ---- accuracy helpers vs a reference (PF ground truth or another solution) ----

"per-bus complex voltage error of a literal `result` against a PMD solution dict"
function voltage_errors(result::LiteralResult, ref_solution::Dict; include_neutral::Bool = true)
    lm_sol = result.solution["bus"]
    errs = Float64[]
    for (b, bsol) in lm_sol
        refb = ref_solution["bus"][b]
        terms = haskey(refb, "terminals") ? refb["terminals"] : 1:length(bsol["vr"])
        for idx in 1:length(bsol["vr"])
            !include_neutral && idx == _N_IDX && continue
            push!(errs, abs((bsol["vr"][idx] + im * bsol["vi"][idx]) -
                            (refb["vr"][idx] + im * refb["vi"][idx])))
        end
    end
    return errs
end

"summary accuracy metrics (RMSE, max |U| error) of a literal result vs ground truth"
function accuracy_metrics(result::LiteralResult, ref_solution::Dict; include_neutral::Bool = true)
    e = voltage_errors(result, ref_solution; include_neutral = include_neutral)
    return (rmse = sqrt(sum(abs2, e) / length(e)), maxerr = maximum(e), n = length(e))
end
