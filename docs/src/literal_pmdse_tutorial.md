# Literal PMDSE tutorial — the PowerGridModel solve options, step by step

This is the recommended **entry point** to the Literal PMDSE estimator. It walks,
matrix by matrix, through the two [PowerGridModel](https://github.com/PowerGridModel/power-grid-model)
(PGM) state-estimation *solve options* reproduced here —

* `:iterative_linear` (PGM's default) and
* `:newton_raphson`

— first on a small **single-phase PGM benchmark** (so we can check every number
against PowerGridModel itself), and then on a **four-wire, explicit-neutral**
network (which PGM cannot represent) using the *same* code. Every intermediate
object is printed: the per-unit data dictionary, the nodal admittance matrix
`Y_bus`, the measurement model, the iterative-linear measurement matrix, the
per-iteration voltage updates, and the Newton–Raphson gain.

The runnable companion script is `examples/literal_pmdse_pgm_tutorial.jl`:

```julia
julia --project examples/literal_pmdse_pgm_tutorial.jl
```

All matrices below are the *actual* output of that script.

---

## Part A — a 3-bus single-phase PGM example

We reproduce the `1os2msr` case from
`power-grid-model/tests/data/state_estimation/1os2msr`: three 10.5 kV nodes, two
lines, a source at node 1, and loads at nodes 2 and 3, with three voltage-phasor
sensors and seven power sensors.

### A.1 Per-unit bases

PGM solves in SI but normalises measurements internally; Literal PMDSE (like
PowerModelsDistribution) works in per-unit. We pick the **same bases PGM uses**
so the weighting — and therefore the estimate — matches exactly:

```
V_base = u_rated = 10500 V        S_base = base_power_3p = 1e6 VA
Z_base = V_base² / S_base ≈ 110.25 Ω
```

A line's PGM parameters `(r1, x1, c1, tan1)` become a PMD branch with
`br_r = r1/Z_base`, `br_x = x1/Z_base`, and a Π-shunt split half to each end,
`b = ω·c1·Z_base/2`, `g = tan1·b`.

### A.2 The nodal admittance matrix

`LiteralModel` assembles `Y_bus` (here, one conductor per bus; node order
`[(1,1), (2,1), (3,1)]`). The nodal current injection is `I = Y_bus · U`:

```
Y_bus  [p.u.]  (3×3)
    457.72-498.53im   -457.64+499.25im          0.0+0.0im
   -457.64+499.25im    801.63-864.33im     -343.81+366.74im
          0.0+0.0im   -343.81+366.74im      343.91-365.8im
```

(The large magnitudes come from the small per-unit line impedances,
`r ≈ 1e-3` p.u.)

### A.3 The measurement model — one "atom" per scalar measurement

`build_se_atoms(lm, math)` turns `data["meas"]` (the same dictionary the JuMP
estimators consume) into a flat list of atoms. Each atom carries the real/imag
parts of two complex coefficient vectors over the nodal voltage `U`:

* `ΔU(x) = cu·U` — a phase-to-neutral voltage (for voltage measurements and as the
  `U` in `S = ΔU·conj(I)`),
* `I(x)  = ci·U` — a branch current row, or a `Y_bus` row for a nodal injection.

For `1os2msr` we get 13 atoms:

```
vim    z=-0.0181  σ=0.01      (voltage angle/imag, node 1)
vre    z=1.0238   σ=0.01      (voltage real, node 1)
vim    z=-0.0134  σ=0.01      (node 2)
vim    z=-0.0207  σ=0.0095    (node 3)
vre    z=1.0234   σ=0.0095    (node 3)
vre    z=1.0239   σ=0.01      (node 2)
power  P=-2.0    Q=1.0    σ=1.0e9    (branch 4, to-side — disabled: σ huge)
power  P=1.2304  Q=-1.7422 σ=0.0209  (branch 5, from)
power  P=2.4124  Q=-3.024  σ=0.0379  (branch 4, from)
power  P=-1.02   Q=-0.22  σ=0.0104   (branch 5, to)
power  P=-1.01   Q=-0.21  σ=0.0103   (load injection, node 2)
power  P=-1.02   Q=-0.22  σ=0.0104   (load injection, node 3)
power  P=2.4124  Q=-3.024  σ=0.038   (source injection, node 1)
```

A voltage phasor sensor becomes two atoms (`:vre`, `:vim`); a power sensor becomes
one `:power` atom (`P`, `Q` with independent σ). Multiple sensors on the same
appliance are Kalman-combined; the disabled to-side sensor (σ = 1e9 p.u.) carries
essentially zero weight.

### A.4 iterative_linear — the linear system it solves

Each atom contributes one or two **real, linear** rows in the state
`x = [vr₁ vr₂ vr₃ vi₁ vi₂ vi₃]`. Stacking them gives the (constant) measurement
matrix `A` (20 rows here). The first eight rows:

```
A  (first 8 of 20 rows)
        0.0      0.0      0.0      0.0      1.0      0.0     # vim node1  -> vi₁
        1.0      0.0      0.0     -0.0     -0.0     -0.0     # vre node1  -> vr₁
        0.0      0.0      0.0      1.0      0.0      0.0     # vim node2
        0.0      0.0      0.0      0.0      0.0      1.0     # vim node3
        0.0      0.0      1.0     -0.0     -0.0     -0.0     # vre node3
        0.0      1.0      0.0     -0.0     -0.0     -0.0     # vre node2
   -457.64   457.72      0.0  -499.25   498.53     -0.0     # Re branch-4 current
    499.25  -498.53      0.0  -457.64   457.72      0.0     # Im branch-4 current
```

The voltage rows are trivial selectors; the power/current rows are rows of
`Y_bus`. `iterative_linear` keeps `A` and the weights `W = diag(1/σ²)` constant
and only re-builds the right-hand side `b` each iteration (the power
measurements are linearised to equivalent currents `I = conj(S/ΔU)` using the
previous voltages), solving `min ‖√W (A x − b)‖₂` each time.

### A.5 Watch it converge

`max_i |Uᵢ − Uᵢ_prev|` shrinks geometrically (ratio ≈ 0.23):

```
it  1  max|ΔU| = 3.3e-2
it  2          = 6.7e-3
it  3          = 1.5e-3
it  5          = 7.8e-5
it 10          = 4.8e-8
it 16          = 6.7e-12   ->  converged, weighted SSR = 0.0
```

(The data is consistent — generated from a power flow — so the residual sum is
machine-zero; the disabled sensor is the only nonzero residual.)

### A.6 newton_raphson — the Gauss–Newton gain

`:newton_raphson` solves the same WLS by Gauss–Newton on the nonlinear model
`z = h(x)`, with `H = ∂h/∂x` (via `ForwardDiff`). It is warm-started from one
`iterative_linear` solve, so it converges in a single refining step here. The
normal-equation gain `G = HᵀWH` (6×6):

```
G = HᵀWH       (entries ~1e10 — the per-unit impedances make G ill-conditioned;
                the QR / orthogonal fallback in the solver handles it)
   5.19e9  -8.53e9   3.35e9   7.21e6   4.54e7  -3.76e7
  -8.53e9   1.98e10 -1.13e10 -4.53e7  -7.73e6   4.28e7
   3.35e9  -1.13e10  7.93e9   3.76e7  -3.20e7  -1.03e7
   7.21e6  -4.53e7   3.76e7   5.19e9  -8.54e9   3.35e9
   4.54e7  -7.73e6  -3.20e7  -8.54e9   1.99e10 -1.13e10
  -3.76e7   4.28e7  -1.03e7   3.35e9  -1.13e10  7.97e9
```

### A.7 Compare to PowerGridModel

Both methods reproduce PowerGridModel's published `sym_output.json` voltages:

| node | PGM \|U\| [V] | iterative_linear \|U\| | newton_raphson \|U\| | max \|U\| error |
|---|---|---|---|---|
| 1 | 10751.073 | 10751.073 | 10751.073 | 6.5e-8 V |
| 2 | 10752.699 | 10752.699 | 10752.699 | 6.5e-8 V |
| 3 | 10748.321 | 10748.321 | 10748.321 | 6.4e-8 V |

The line flows (including the to-side back-calculation) and the disabled
sensor's residual also match PGM — this is gate 6 of the test suite
(`test/literal/test_pgm_benchmark.jl`).

---

## Part B — the same solver, four-wire with an explicit neutral

PGM is single-phase (Kron-reduced). Literal PMDSE keeps the neutral explicit:
every bus has terminals `[1,2,3,4]`, voltages are phase-to-neutral, and a wye
load's neutral current is `−Σ(phase currents)`. The *same* `build_se_atoms` /
`solve_se_*` code handles it — the single-phase case above is just the
one-conductor, no-neutral special case.

### B.1 The 4×4 mutually-coupled branch

```
Y_branch = Z⁻¹  [p.u.]   (row/col 4 = neutral)
     8.07-16.1im    -1.93+3.9im    -1.93+3.9im   -0.71+0.98im
    -1.93+3.9im      8.07-16.1im   -1.93+3.9im   -0.71+0.98im
    -1.93+3.9im     -1.93+3.9im     8.07-16.1im  -0.71+0.98im
   -0.71+0.98im    -0.71+0.98im    -0.71+0.98im   7.09-9.79im
```

### B.2 The neutral return current

With per-phase current-injection measurements at the load, the estimator
synthesises the **neutral** nodal-injection atom as minus the sum of the phase
currents:

```
cinj at node (2,1)   Ir=-0.298  Ii= 0.102
cinj at node (2,2)   Ir= 0.311  Ii= 0.292
cinj at node (2,3)   Ir= 0.106  Ii=-0.523
cinj at node (2,4)   Ir=-0.120  Ii= 0.129     #  = −(−0.298+0.311+0.106) , −(0.102+0.292−0.523)
```

### B.3 Recovery, neutral included

Both methods recover the power-flow truth to machine precision:

```
ITERATIVE_LINEAR: converged in 2 it,  max|U − truth| = 5.55e-16
NEWTON_RAPHSON:   converged in 1 it,  max|U − truth| = 5.66e-16
bus-2 voltages |U| per terminal [1,2,3,N] = [0.99191, 0.98553, 0.98281, 0.01322]
```

The non-zero neutral voltage (`0.01322` p.u.) is exactly the unbalance effect a
four-wire study needs and that a Kron-reduced single-phase model cannot show.
This is gate 7 of the test suite (`test/literal/test_pgm_en_generality.jl`).

---

## Part C — reference schemes and the robust estimators

The reference scheme fixes the rotational gauge of the WLS problem:

| `reference` | effect |
|---|---|
| `:prop` | fix only grounded terminals; the absolute angle is taken from a voltage-phasor measurement, or pinned automatically (`Im U_ref = 0`) when only magnitudes are measured. |
| `:sota` | additionally ground the reference-bus neutral and fix one angle datum. |
| `:full_slack` | fix the whole reference-bus phasor (e.g. to a known slack). |

```
reference schemes (method=:iterative_linear):  :prop -> converged (13 it)
                                               :sota -> converged (13 it)
```

Besides the two PGM solve options, the original JuMP-free **Gauss–Newton**
estimators remain available (leave `method` unset). They target the
explicit-neutral IVREN model:

| `estimator` | meaning |
|---|---|
| `:wls` | weighted least squares (Gauss–Newton with a QR fallback) |
| `:wlav` | weighted least *absolute value* (IRLS) — robust to a single gross error |
| `:mle` | general maximum likelihood (Fisher scoring) — reduces to `:wls` for Gaussian noise |

```
Gauss-Newton estimators on the four-wire EN network:  :wls / :wlav / :mle -> all converged
```

---

## API summary

```julia
using PowerModelsDistributionStateEstimation

# the two PowerGridModel solve options (general — single-phase or four-wire EN):
res = solve_mc_se_literal(data; estimator=:wls, method=:iterative_linear)
res = solve_mc_se_literal(data; estimator=:wls, method=:newton_raphson)

# the original Gauss-Newton estimators (explicit-neutral IVREN model):
res = solve_mc_se_literal(data; estimator=:wls)         # or :wlav, :mle

# lower-level introspection:
lm    = LiteralModel(data; reference=:prop)
atoms = build_se_atoms(lm, data)                        # the measurement model
res   = solve_se_il(lm, atoms)                          # iterative_linear
res   = solve_se_nr(lm, atoms)                          # newton_raphson
```

`res` is a [`LiteralResult`](@ref): `res.solution["bus"][id]["vr"|"vi"|"vm"]`,
`res.termination`, `res.iterations`, `res.objective`, `res.gain_cond`,
`res.gain_rank`. See [Literal PMDSE (matrix-based)](@ref) for the full reference.
