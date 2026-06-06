# Literal PMDSE — explicit matrix-based explicit-neutral DSSE

**Status: complete — all 5 validation gates pass.**

`Literal PMDSE` is a textbook, JuMP-free distribution-system state estimator that
consumes the *same* PMD mathematical dictionaries (`data_math`,
`data_math["meas"]`, `data_math["se_settings"]`) as the JuMP IVR explicit-neutral
estimator (`solve_ivr_en_mc_se` / `IVRENPowerModel`, `src/prob/se_en.jl`) but
solves the DSSE by forming the measurement model `z = h(x) + e`, the Jacobian
`H = ∂h/∂x` (via `ForwardDiff`) and iterating the normal equations
`(HᵀWH)Δx = HᵀW r` (Abur & Expósito, *Power System State Estimation*).

```julia
res = solve_mc_se_literal(data_math; estimator = :wls,  reference = :sota)
res = solve_mc_se_literal(data_math; estimator = :wlav, reference = :full_slack, ref_values = …)
res = solve_mc_se_literal(data_math; estimator = :mle)        # Gaussian MLE == WLS
```

## Files

```
src/bare/
  literal_core.jl       # node index map, Ybus assembly, references, flat start, solution dict
  measurement_model.jl  # rectangular EN h(x) closures for every measurement var-type
  solve_wls.jl          # Gauss–Newton WLS (normal equations + QR/orthogonal fallback)
  solve_wlav.jl         # IRLS WLAV (robust to gross errors)
  mle.jl                # general log-likelihood interface (Newton); Gaussian reduces to WLS
  literal_pmdse.jl      # solve_mc_se_literal wrapper + LiteralResult + accuracy helpers
test/literal/
  helpers.jl            # toy network, toy PF, feeder loaders, measurement builders
  test_ybus.jl          # gate 1
  test_hx_jacobian.jl   # gate 2
  test_references.jl    # gate 3
  test_wls_vs_ivren.jl  # gate 4
  test_benchmark.jl     # gate 5
```

## Formulation (nodal, rectangular, explicit-neutral)

* **State** `x = [vr; vi]` over every `(bus, terminal)` pair, terminals including
  the neutral `_N_IDX = 4`.  Branch / load / gen currents are eliminated.
* **Admittance** `Ybus = G + jB`: each branch contributes its full mutually
  coupled series admittance `Y = (br_r + j·br_x)^{-1}` stamped over
  `f_connections`/`t_connections`, plus its Π line-shunts (`g_fr,b_fr,g_to,b_to`)
  and any bus shunts.  Validated to reproduce PMD's power-flow current balance
  (`Ybus·U == Σgen − Σload` at every node).
* **Nodal injection** `I = Ybus·U`, evaluated `Ir = G·vr − B·vi`,
  `Ii = B·vr + G·vi` (ForwardDiff-friendly).
* **Measurement functions** (EN, phase-to-neutral where relevant):
  `vr,vi` identity; `vm/vmn = |U_c−U_n|`; `va = ∠(U_c−U_n)`; `vll = |U_i−U_j|`;
  branch `cr,ci,cm,ca` from `Y_ff·U_f + Y_ft·U_t`; branch power `p,q` (EN);
  current injections `crd,cid,crg,cig` and powers `pd,qd,pg,qg,ptot,qtot` and
  `cmd,cmg,cad,cag` via the nodal injection `Ybus·U`.

## Key modelling decisions

1. **Component → nodal aggregation.**  Current/power *injection* measurements are
   mapped to the nodal injection `Ybus·U`.  When several components share a
   bus terminal (e.g. the 4 loads on one bus of `3bus_4wire`), their measured
   currents are summed into one nodal residual — exact at the solution and the
   only consistent choice once currents are eliminated.  A wye component's
   **neutral injection is `−Σ(phase injections)`**.
2. **Zero-injection pseudo-measurements** (`z = 0`, high weight) close the model
   at source-free terminals, reproducing the KCL equalities of `IVRENPowerModel`
   and making intermediate buses observable.
3. **References (configurable).**  `bus_type == 3` is the reference bus.
   * `:sota` (default) — ground the reference-bus neutral **and** fix one angle
     datum `vi[ref, phase₁] = 0`.  This is the SOTA combination from `main.tex`
     that makes the gain full-rank without forcing a balanced reference bus.
   * `:full_slack` — fix the whole reference-bus phasor (used for the apples-to-
     apples gate-4 comparison against IVREN).
   * `:prop` — neutral only; **rank-deficient by one** (the observability trap).
   Grounded terminals (`bus["grounded"]`) are always fixed to 0.
4. **Numerics.**  Normal equations with a Cholesky solve, falling back to the
   orthogonal QR least-squares on `√W·H` when the gain is ill-conditioned (the
   per-unit impedances of `3bus_4wire` give `cond(G) ~ 1e17`, yet the QR fallback
   recovers the state to machine precision).
5. **Weights.**  `W = diag(1/(rescaler·σ)²)` with `σ = std(dst)`; `Float64`
   `dst` entries are treated as near-exact (floored σ); a σ-floor prevents an
   infinite weight when a measured value is exactly 0 (e.g. `vi` at a balanced
   reference bus).
6. **MLE-ready.**  `solve_mle` maximises `Σ logpdf(dst_i, h_i(x))` by a damped
   Newton iteration; for Gaussian `dst` it is identical to WLS (verified to
   ~1e-16).  An arbitrary `Distributions`/`Polynomials`/`ExtendedBeta` can be
   supplied with no change to `h(x)` or the references.

## Validation gates (all pass)

| # | Gate | Result |
|---|------|--------|
| 1 | `Ybus·U` reproduces PMD's nodal current balance | max err `~1e-10` |
| 2 | AD `H` vs finite differences, every measurement type | rel err `<1e-6` |
| 3 | rotation unobservable w/o angle datum (`:prop`), observable with it (`:sota`) | `‖H·δx_rot‖ ~1e-17` vs `5e-2` |
| 4 | noiseless literal WLS == IVREN / PF state | max\|U\| `~1e-16` (`:full_slack` & `:sota`); `<1e-4` vs JuMP IVREN |
| 5 | benchmark IVREN vs literal WLS/WLAV (accuracy + time/iters) | see table |

### Benchmark (3-bus 4-wire, σ = 0.01, 4 noise seeds)

| seed | IVREN max\|U\| / t[s] | WLS max\|U\| / t[s] / it | WLAV max\|U\| / t[s] / it |
|------|----------------------|--------------------------|---------------------------|
| 1 | 6.9e-4 / 0.52 | 2.8e-4 / 0.001 / 2 | 2.5e-4 / 1.45 / 7 |
| 2 | 8.6e-4 / 0.45 | 7.6e-4 / 0.001 / 2 | 6.8e-4 / 0.001 / 8 |
| 3 | 3.6e-4 / 0.45 | 1.4e-3 / 0.001 / 2 | 1.3e-3 / 0.001 / 6 |
| 11 | 4.4e-4 / 0.49 | 2.1e-4 / 0.001 / 2 | 2.0e-4 / 0.001 / 6 |
| **mean** | **5.9e-4** | **6.6e-4** | **6.1e-4** |

The literal Gauss–Newton WLS converges in **2 iterations / ~1 ms**, roughly
**500× faster** than the JuMP/Ipopt IVREN solve, with comparable accuracy;
WLAV is marginally more accurate (robustness).  Results are *compared*, not
asserted equal.

## Sign / convention notes (verified against IVREN)

* Nodal injection `Ybus·U = Σ crg − Σ crd` (gen `+`, load `−`).
* Power injection `p_c = s·(Ir_c·Δvr_c + Ii_c·Δvi_c)`, `q_c = s·(−Ii_c·Δvr_c +
  Ir_c·Δvi_c)`, `Δ` phase-to-neutral, `s = +1` gen / `−1` load.
* Branch from-current `I_fr = (Y_series+Y_sh_fr)·U_f − Y_series·U_t`; branch flow
  `p = cr·Δvr_f + ci·Δvi_f` (EN).

## Limitations / future work

* Transformers are not yet folded into `Ybus` (the explicit-neutral test feeders
  use a stiff voltage source, no in-service transformer); branches + shunts are
  supported.
* Power-*only* injection buses rely on the power residuals for observability of
  that node; the canonical IVREN set uses current injections.
* `calc_admittance_matrix` in PMD does not support the 4-wire (kron_reduce=false)
  case, so gate 1 validates `Ybus` against PMD's power-flow physics instead.
* Optional analytic Jacobian blocks (unit-tested against AD) could replace
  `ForwardDiff` for speed on large feeders.
