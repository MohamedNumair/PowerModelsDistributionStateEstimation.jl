# Literal PMDSE — explicit matrix-based estimator

As of version 0.8.0, PMDSE ships **Literal PMDSE**: a textbook, **JuMP-free**
distribution-system state estimator that mirrors the 4-wire IVR explicit-neutral
model (`IVRENPowerModel`, see [Explicit Neutral Models for DSSE](@ref))
but solves the DSSE with **explicit matrices** — the measurement model
`z = h(x) + e`, the Jacobian `H = ∂h/∂x`, the gain `G = HᵀWH` and the normal
equations `G Δx = HᵀW r` — instead of handing a nonlinear program to a solver
such as Ipopt.

It consumes the **same** PMD mathematical data dictionary as
`solve_ivr_en_mc_se` (`data_math`, `data_math["meas"]`, `data_math["se_settings"]`),
so it is a drop-in alternative whenever you want a fast, transparent Gauss–Newton
estimate, a robust WLAV estimate, or a general maximum-likelihood estimate, and
full control over the angular/zero-sequence reference (see
[How Literal PMDSE resolves the angular & zero-sequence reference problem](@ref)).

## Quick start

```julia
import Ipopt
import PowerModelsDistribution as _PMD
import PowerModelsDistributionStateEstimation as _PMDSE

# --- parse a 4-wire explicit-neutral feeder and build ground truth ----------
eng = _PMD.parse_file(joinpath(_PMDSE.BASE_DIR, "test", "data",
                               "three-bus-en-models", "3bus_4wire.dss"))
_PMD.transform_loops!(eng)
_PMD.remove_all_bounds!(eng)
math = _PMD.transform_data_model(eng, kron_reduce=false, phase_project=false)
_PMD.add_start_vrvi!(math)
pf = _PMD.solve_mc_opf(math, _PMD.IVRENPowerModel, Ipopt.Optimizer)

# --- synthesize the canonical IVREN measurement set (vr/vi + crd/cid + …) ---
msr = joinpath(mktempdir(), "msr.csv")
_PMDSE.write_measurements!(_PMD.IVRENPowerModel, math, pf, msr, σ=0.005)
_PMDSE.add_measurements!(math, msr, actual_meas=true)
math["se_settings"] = Dict{String,Any}("rescaler" => 1.0)

# --- run the literal estimator (no JuMP / no solver object needed) -----------
res = _PMDSE.solve_mc_se_literal(math; estimator=:wls, reference=:sota)

@show res.termination          # :converged
@show res.iterations           # ~2-3 Gauss-Newton iterations
@show res.solution["bus"]["3"]["vm"]   # estimated phase-to-ground magnitudes
```

## `solve_mc_se_literal`

```julia
solve_mc_se_literal(data; estimator = :wls, method = nothing, reference = :sota,
                    ref_values = nothing, maxiter = 50, tol = 1e-9,
                    verbose = false)
```

| argument | values | meaning |
|----------|--------|---------|
| `data` | `Dict` | PMD **mathematical** dictionary with `data["meas"]` populated (see [Measurement Conversion](@ref)) and an optional `data["se_settings"]`. |
| `estimator` | `:wls` (default), `:wlav`, `:mle` | which estimator to run (see below). |
| `method` | `nothing` (default), `:iterative_linear`, `:newton_raphson` | for `estimator = :wls`, selects a PowerGridModel solve option (see below); `nothing` keeps the in-house Gauss–Newton WLS. |
| `reference` | `:sota` (default), `:full_slack`, `:prop` | reference / observability scheme (see below). |
| `ref_values` | `(vr::Dict, vi::Dict)` or `nothing` | reference-bus phasor (keyed by terminal) for `:full_slack`. |
| `maxiter` | `Int` | maximum Gauss-Newton / IRLS / Newton iterations. |
| `tol` | `Float64` | convergence tolerance on `‖Δx‖∞`. |
| `verbose` | `Bool` | print per-iteration progress. |

`data["se_settings"]` is honoured: `"rescaler"` scales the weights
(`W = diag(1/(rescaler·σ)²)`, consistent with `constraint_mc_residual`), and an
optional `"reference"` entry overrides the `reference` keyword.

### Estimators

| `estimator` | method | use when |
|-------------|--------|----------|
| `:wls`  | Gauss–Newton **Weighted Least Squares** (normal equations with a QR / orthogonal fallback for ill-conditioned gains) | the default; fastest, optimal for Gaussian noise. |
| `:wlav` | **Weighted Least Absolute Value** via IRLS | robustness to a single gross/bad measurement. |
| `:mle`  | general **Maximum Likelihood** by Fisher scoring on `Σ logpdf(dstᵢ, hᵢ(x))` | non-Gaussian measurement errors; reduces **exactly** to WLS for Gaussian `dst`. |

### PowerGridModel solve options (`method`)

With `estimator = :wls` the two
[PowerGridModel](https://github.com/PowerGridModel/power-grid-model) WLS *solve
options* can be selected through `method`. Both are general (four-wire,
explicit-neutral capable — the single-phase PGM network is the one-conductor,
no-neutral special case) and are validated against PowerGridModel's own
state-estimation examples (test gates 6–7).

| `method` | algorithm | notes |
|----------|-----------|-------|
| `:iterative_linear` | measurements linearised to complex currents / voltage-phasors at the previous voltages; a constant linear WLS system re-solved each iteration | PowerGridModel's default; the slack-angle gauge is pinned automatically when no voltage angle is measured. |
| `:newton_raphson`   | Gauss–Newton on the nonlinear WLS, warm-started from one `iterative_linear` solve | identical optimum to `:iterative_linear`; warm-start avoids the flat-start degeneracy of power-/magnitude-only systems. |

```julia
res = solve_mc_se_literal(data; estimator = :wls, method = :iterative_linear)
res = solve_mc_se_literal(data; estimator = :wls, method = :newton_raphson)
```

```julia
res_wls  = _PMDSE.solve_mc_se_literal(math; estimator=:wls)
res_wlav = _PMDSE.solve_mc_se_literal(math; estimator=:wlav)   # rejects gross errors
res_mle  = _PMDSE.solve_mc_se_literal(math; estimator=:mle)    # == WLS for Gaussian dst
```

### Reference schemes

The reference bus is the bus with `bus_type == 3`. Grounded terminals
(`bus["grounded"]`) are always fixed to `0`. On top of that:

| `reference` | fixes | rank | use |
|-------------|-------|------|-----|
| `:sota` (default) | reference-bus neutral `vr=vi=0` **and** one angle datum `vi[ref, phase₁]=0` | full | the SOTA combination — observable **without** forcing a balanced reference bus. |
| `:full_slack` | the whole reference-bus phasor (`ref_values`, or balanced if omitted) | full | apples-to-apples comparison against a fixed-slack solver. |
| `:prop` | reference-bus neutral only | **deficient by 1** | the "neutral-only" proposition — *unobservable*; for observability studies. |

```julia
# SOTA: ground neutral + one angle datum (estimates the unbalanced reference bus)
res = _PMDSE.solve_mc_se_literal(math; estimator=:wls, reference=:sota)

# Fix the full reference-bus phasor to known (e.g. measured / true) values
rb   = _PMDSE.literal_ref_bus(math)
bsol = pf["solution"]["bus"]["$rb"]; terms = math["bus"]["$rb"]["terminals"]
rv   = (Dict(t=>bsol["vr"][i] for (i,t) in enumerate(terms)),
        Dict(t=>bsol["vi"][i] for (i,t) in enumerate(terms)))
res  = _PMDSE.solve_mc_se_literal(math; reference=:full_slack, ref_values=rv)
```

See [How Literal PMDSE resolves the angular & zero-sequence reference problem](@ref)
for the theory behind `:prop` vs `:sota`.

## The result object

`solve_mc_se_literal` returns a `LiteralResult`:

| field | description |
|-------|-------------|
| `termination` | `:converged` or `:maxiter` |
| `iterations`  | number of iterations |
| `objective`   | weighted SSR `rᵀWr` (WLS/MLE) or `Σ√wᵢ|rᵢ|` (WLAV) at the solution |
| `solve_time`  | wall-clock seconds |
| `gain_cond`, `gain_rank` | condition number / numerical rank of `G = HᵀWH` (health check) |
| `solution`    | `Dict("bus" => id => Dict("vr","vi","vm"))`, vectors ordered like `bus["terminals"]` |
| `x_free`, `x_full` | the estimated state (free entries / full `[vr; vi]`) |
| `estimator`   | `:wls`, `:wlav` or `:mle` |

### Accuracy vs ground truth

```julia
m = _PMDSE.accuracy_metrics(res, pf["solution"])      # (rmse, maxerr, n) over all |U|
m = _PMDSE.accuracy_metrics(res, pf["solution"]; include_neutral=false)  # phases only
e = _PMDSE.voltage_errors(res, pf["solution"])        # per-node |ΔU| vector (incl. NEV)
```

## Supported measurements

Literal PMDSE implements `h(x)` for every measurement var-type that
`IVRENPowerModel` supports (see [Measurement Conversion](@ref)):

* **native**: `:vr, :vi, :cr, :ci, :crd, :cid, :crg, :cig`
* **voltage**: `:vm`/`:vmn` (phase-to-neutral), `:va`, `:vll`
* **branch**: `:cm, :ca, :p, :q`
* **injection**: `:pd, :qd, :pg, :qg, :ptot, :qtot, :cmd, :cmg, :cad, :cag`

Voltage and power measurements follow the EN phase-to-neutral convention and the
neutral residual row is skipped (`setdiff(conns, _N_IDX)`), exactly as in
`constraint_mc_residual`. Current/power *injection* measurements are mapped to the
nodal injection `Ybus·U`; when several components share a bus terminal (e.g.
multiple loads on one bus) their measurements are aggregated into one nodal
residual — exact at the solution.

## Lower-level API (advanced)

For custom workflows you can build and reuse the pieces directly:

```julia
lm    = _PMDSE.LiteralModel(math; reference=:sota)   # node map, Ybus, references
model = _PMDSE.build_se_model(lm, math; rescaler=1.0)# residual rows z, w, h-closures
res   = _PMDSE.solve_wls(model)                       # or solve_wlav / solve_mle

H = ForwardDiff.jacobian(x -> _PMDSE.predict(model, x), res.x_free)  # the Jacobian
```

`build_se_model` accepts `rescaler`, `σ_floor` (prevents an infinite weight when a
measured value is exactly 0, e.g. `vi` at a balanced reference bus) and
`zero_inj_sigma` (weight of the zero-injection pseudo-measurements).

## Notes & limitations

* Transformers are not yet folded into `Ybus` (the explicit-neutral test feeders
  use a stiff voltage source); mutually-coupled branches and shunts are supported.
* PMD's `calc_admittance_matrix` does not support the 4-wire
  (`kron_reduce=false`) case, so `Ybus` is validated against PMD's power-flow
  current balance in the test suite.
* The Jacobian is assembled with `ForwardDiff`, keeping the estimation algebra
  fully explicit while avoiding hand-derivative bugs.
