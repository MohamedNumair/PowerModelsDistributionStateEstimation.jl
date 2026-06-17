# Session handoff — Literal PMDSE × PowerGridModel

A self-contained record of the work done in this session so any future session
(which starts from a fresh clone) can pick it up and reason about it.

- **Repo:** `MohamedNumair/PowerModelsDistributionStateEstimation.jl`
- **Branch:** `claude/gallant-fermi-zfg03q` (this is the working copy of the
  `Literal_PMDSE` feature branch; it was branched from, and is what feeds, the
  `Literal_PMDSE` work)
- **Open PR:** #3 — <https://github.com/MohamedNumair/PowerModelsDistributionStateEstimation.jl/pull/3>
- **Benchmark source repo (read-only):** `MohamedNumair/power-grid-model`
  (a clone of LF Energy's `power-grid-model`); left **unmodified**.
- **HEAD at handoff:** `90624a7`; two commits beyond `Literal_PMDSE`:
  - `685968a` — PGM WLS solve options + validation gates
  - `90624a7` — verbose PGM tutorial (entry point)

---

## 1. The task

Enhance the `Literal_PMDSE` branch of PowerModelsDistributionStateEstimation.jl
(a textbook, JuMP-free distribution state estimator) to:

1. Implement the two **PowerGridModel (PGM)** state-estimation *solve options*
   — `iterative_linear` and `newton_raphson` — and validate "the same
   mathematical solve options".
2. Use **PGM's own examples as the benchmark** (TDD: pick solved PGM SE cases,
   encode them as tests).
3. Keep the implementation **general / four-wire (explicit neutral)** even though
   PGM is single-phase / Kron-reduced.
4. Add **verbose tutorials** (matrices and intermediate steps) as the entry point.
5. Keep everything **consistent with the package's data import** (the PMD
   `data_math` dict + `data["meas"]`).

Background context the user supplied: a long technical reference on how PGM does
WLS state estimation (the two methods, the augmented/Hachtel form, per-unit
normalisation, σ_S² = σ_P²+σ_Q², slack-angle gauge, etc.).

---

## 2. What was built

### New solver module — `src/bare/pgm_se.jl` (~600 lines)

A general, four-wire-capable re-implementation of PGM's two WLS solve options on
the literal nodal model (state `x = [vr; vi]` over every `(bus, terminal)`).

- `SEAtom` — one scalar measurement, stored as the real/imag parts of two complex
  coefficient vectors over the nodal voltage `U`:
  - `cu` → `ΔU(x) = cu·U` (phase-to-neutral voltage),
  - `ci` → `I(x) = ci·U` (a branch current row, or a `Y_bus` row for an injection).
  - kinds: `:vre`, `:vim` (voltage phasor components), `:vmag` (magnitude),
    `:power` (P,Q of `S = sign·ΔU·conj(I)`), `:cinj` (complex current),
    `:zinj` (zero-injection KCL).
- `build_se_atoms(lm, math; rescaler)` — parses `math["meas"]` (the package
  measurement dict) into atoms. Handles: phase-to-neutral voltages, branch
  power/current (from/to side via `meta["side"]`), appliance injection
  **aggregation** onto the bus, the **neutral return current = −Σ phase
  currents**, **Kalman-combine** of repeated sensors on the same quantity
  (PGM §3.3), and zero-injection pseudo-measurements.
- `solve_se_il(lm, atoms; ...)` — **iterative_linear**: constant real
  measurement matrix `A`, weights `W = diag(1/σ²)`, RHS re-linearised each
  iteration (`I = conj(S/ΔU)` at previous voltages); solves
  `min ‖√W (A x − b)‖₂` (QR). Power weight uses `1/(σp²+σq²)` (PGM combines).
- `solve_se_nr(lm, atoms; warm=true)` — **newton_raphson**: Gauss–Newton on the
  nonlinear `z = h(x)` with `H = ∂h/∂x` via `ForwardDiff`, **warm-started from one
  `iterative_linear` solve** (the flat start is degenerate for
  power-/magnitude-only systems — the angle Jacobian vanishes). NR power weights
  are independent `1/σp²`, `1/σq²`.
- `_gauge(lm, atoms)` — reference/gauge: respects `lm.fixed_mask`; if **no**
  voltage angle/phasor is measured, pins `Im(U_ref)=0` automatically.
- Reuses `LiteralModel` (Y-bus, node map, references), `LiteralResult`,
  `Symmetric_full`, `_solve_gain`, `_gain_diag` (from `literal_core.jl` /
  `solve_wls.jl`) and `_zsigma`, `_nonneutral`, `_active_connections` (from
  `measurement_model.jl`).

### API wiring

- `solve_mc_se_literal(data; estimator=:wls, method=nothing, …)` gained a
  `method` kwarg (`src/bare/literal_pmdse.jl`):
  - `method === nothing` → the original Gauss–Newton `solve_wls` (unchanged,
    backward compatible; this path is the EN-oriented `build_se_model` one).
  - `method = :iterative_linear` → `solve_se_il`
  - `method = :newton_raphson` → `solve_se_nr`
  - also reads `data["se_settings"]["method"]` if present.
- Exports added (`src/core/export.jl`): `build_se_atoms`, `solve_se_il`,
  `solve_se_nr`, `SEAtom`.
- Included after `solve_wls.jl` in the main module so its dependencies exist.

### Tests (TDD gates 6 & 7)

- `test/literal/pgm_cases.jl` — embedded golden cases (`const PGM_CASES`) +
  `build_pgm_math`/`build_pgm_meas!` (PGM network → per-unit PMD `data_math` +
  `data["meas"]`) + `pgm_result_voltage`.
- `test/literal/test_pgm_benchmark.jl` — **gate 6**: literal IL & NR reproduce
  PGM's published node voltages **and line flows** on `single-node`, `1os2msr`,
  `1os2msr-no-angle`, `single-line-load-il`; asserts the two methods agree.
- `test/literal/test_pgm_en_generality.jl` — **gate 7**: the same solve options
  run on the explicit-neutral `3bus_4wire` feeder, recover the IVREN power flow
  (neutral incl.), and agree with `solve_wls`.
- `test/literal/pgm/` — `generate_golden.py` (portable, `python3
  generate_golden.py /path/to/power-grid-model`), `golden.json`,
  `pgm_cases_data.jl` (regenerated block), `README.md`.
- `test/runtests.jl` — registers gates 6 & 7.

### Tutorials (entry point)

- `examples/literal_pmdse_pgm_tutorial.jl` — runnable, verbose; prints every
  intermediate (per-unit bases, `Y_bus`, atoms, the IL `A` matrix, per-iteration
  `|ΔU|`, the NR gain `HᵀWH`, the PGM comparison, then a four-wire EN example with
  the 4×4 coupled admittance and the neutral return, then reference schemes /
  estimators).
- `docs/src/literal_pmdse_tutorial.md` — narrated version with the **real
  captured matrices**; registered in `docs/make.jl`.
- `docs/src/literal_pmdse.md`, `docs/src/Literal_PMDSE_PLAN.md`, `CHANGELOG.md`
  updated.

---

## 3. Key technical decisions (and why)

- **Per-unit with PGM's bases** (`V_base = u_rated`, `S_base = base_power_3p =
  1e6`, `Z_base = u_rated²/1e6`). This is essential: the WLS *weighting* (hence
  the estimate on inconsistent/weighted data, and the IL convergence rate)
  depends on normalisation. Confirmed in PGM source
  (`common/common.hpp: base_power_3p = 1e6`).
- **Line Π-model:** `y_series = 1/(r1+j·x1)`, total shunt `b = ω·c1`,
  `g = tan1·b`, split half to each end. Verified against PGM line flows.
- **NR warm-started from IL.** Rectangular Gauss–Newton from a flat start is
  rank-degenerate for power-/magnitude-only systems (no phasor) — the classic
  flat-start angle-Jacobian collapse. Warm-starting from IL makes NR match PGM
  on every case PGM tests, converging to the identical optimum.
- **Gauge:** with a voltage angle/phasor measurement present, the absolute angle
  comes from it (no extra constraint); otherwise pin `Im(U_ref)=0` (equivalent to
  PGM's per-iteration slack-angle normalisation).
- **Neutral handling (four-wire generality):** voltages are phase-to-neutral
  (`cu = e_kc − e_kn`, or `e_kc` if no neutral); a wye appliance's neutral
  injection current = −Σ phase currents. Single-phase PGM = the no-neutral
  special case.
- **Fixed-column handling in IL:** for a fully-fixed (`:full_slack`) reference
  bus, the fixed contribution is moved to the RHS (`b_eff = b − A·x_fix`). This
  only matters when fixed values are non-zero (EN full-slack); the single-phase
  PGM cases fix nothing, which is why it was easy to miss.

---

## 4. How it was validated (important — sandbox could not run the full suite)

The sandbox **could not install the Julia package registry** (all `*.julialang.org`
hosts are 403 under the environment's "Trusted" network policy; `github.com` is
proxy-restricted to the in-scope repos). So the full `]test` (gates 1–5 + `.dss`
parse + Ipopt) was **not** run here. Three independent checks were used instead;
all green:

1. **PGM oracle (Python).** `pip install power-grid-model` works
   (`pypi.org` allowed). Ran PGM on its own JSON cases to produce golden node
   voltages / line flows / residuals → `test/literal/pgm/golden.json`.
2. **Python "mirror".** A line-by-line Python port of the exact Julia logic
   (meas-dict parsing, atom assembly, Kalman combine, gauge, fixed-column IL,
   warm-started NR) was run against the PGM oracle: single-phase ≤1e-7, and a
   four-wire EN power-flow recovery to ~1e-14 (neutral incl.).
3. **Real Julia (registry-free).** Installed **Julia 1.12.5 via conda-forge**
   (`conda.anaconda.org` is on the Trusted allowlist; bootstrapped `micromamba`
   from a conda package, then `micromamba create -p /opt/julia -c conda-forge
   julia`). Then loaded the **actual** `pgm_se.jl` (+ `literal_core.jl`,
   `measurement_model.jl`, `solve_wls.jl`) into a harness module that stubs only
   the three packages I can't fetch: `Distributions`→a tiny `Normal`,
   `ForwardDiff`→**complex-step AD** (exact for these analytic functions),
   `PowerModelsDistribution`→inert enums (`LinearAlgebra` is a real stdlib).
   Results from running the real code:
   - Gate 6: `single_node` 1.8e-12 V, `1os2msr` IL 8.0e-8 / NR 3.2e-10,
     `1os2msr_no_angle` IL 1.8e-7 / NR 4.2e-10, `single_line_load` IL 9.8e-6;
     IL≡NR ≤1.5e-8 V.
   - Gate 7 (EN recovery): max|U−truth| = 5.6e-16 (neutral incl.).
   - The full tutorial script runs end-to-end (Parts A/B/C).
   - All `.jl` files `Meta.parseall` clean.

> The complete `]test` (gates 1–7) is expected to run in **GitHub Actions on
> PR #3** (CI has unrestricted network) — that is the authoritative full-suite
> check.

### Reproducing the real-Julia harness in a future session

Julia install (works under Trusted):
```bash
curl -fsSL -o /tmp/mm.tar.bz2 https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.8.1-0.tar.bz2
mkdir -p /opt/mm && tar -xjf /tmp/mm.tar.bz2 -C /opt/mm bin/micromamba
MAMBA_ROOT_PREFIX=/opt/conda /opt/mm/bin/micromamba create -y -p /opt/julia -c conda-forge --override-channels julia
/opt/julia/bin/julia --version
```
The harness pattern: a module that `import LinearAlgebra`; defines `_N_IDX=4`,
stub `_DST` (`Normal`+`mean`/`std`), stub `ForwardDiff.jacobian` via complex-step,
stub `_PMD` (`@enum WYE DELTA` / `@enum POWER`); then `include`s the four source
files. Load `test/literal/pgm_cases.jl` after stripping its `import Distributions`
/ `import PowerModelsDistribution` lines. (If the env has full network, skip all
this and just `Pkg.instantiate(); Pkg.test()`.)

---

## 5. Network / environment notes

- This is an ephemeral cloud container; the **network policy is fixed at
  container creation**, so changing the environment's "Network access" applies to
  **new sessions only**. In this session it stayed "Trusted":
  allowed = pypi.org, files.pythonhosted.org, conda.anaconda.org,
  repo.anaconda.com, github.com (scoped via the git proxy); blocked (403) =
  `*.julialang.org`, api.github.com, api.anaconda.org, general internet.
- To run the full Julia suite locally in a session, the env must be set to
  **Full**, or **Custom** + "include defaults" + `*.julialang.org`, and a **fresh
  session started**.

---

## 6. Open items / next steps

- [ ] **Confirm CI is green on PR #3** (the full gates 1–7). If there is no Julia
      workflow under `.github/workflows`, add one so the suite runs on PRs.
- [ ] If a fresh full-network session is available: `Pkg.instantiate(); Pkg.test()`
      and fix anything the stubbed harness couldn't catch (e.g. real `ForwardDiff`
      edge cases — though complex-step is exact for the current `h(x)`; CSV-based
      `add_measurements!` parsing of the benchmark vocabulary; Ipopt PF in gate 7).
- [ ] **Optional generalisation:** the legacy `build_se_model` / `solve_wls`
      path (selected by `method=nothing`, used by `:wlav`/`:mle`) still **assumes
      an explicit neutral** (it `KeyError`s on a single-phase bus). The new
      `pgm_se.jl` path is general. Generalising `build_se_model`'s neutral lookups
      (mirroring `_delta_coeff`) would let `:wlav`/`:mle` run on single-phase too.
      Deferred (low priority, additive, but touches tested code).
- [ ] **Optional:** the four-wire EN benchmark in gate 7 uses the `3bus_4wire.dss`
      feeder + Ipopt; could add a registry-free EN gate (build `data_math`
      in-Julia, Newton PF for truth) so part of the EN validation runs without
      Ipopt.
- [ ] **Watch PR #3 / auto-fix CI** was offered but not yet enabled.

---

## 7. File map (everything added/changed on this branch)

```
src/bare/pgm_se.jl                      NEW  the IL + NR solver, atoms, build_se_atoms
src/bare/literal_pmdse.jl              EDIT  `method` kwarg dispatch
src/PowerModelsDistributionStateEstimation.jl EDIT include pgm_se.jl
src/core/export.jl                     EDIT  export build_se_atoms/solve_se_il/solve_se_nr/SEAtom
test/literal/pgm_cases.jl               NEW  golden data + PGM→PMD builders
test/literal/test_pgm_benchmark.jl      NEW  gate 6
test/literal/test_pgm_en_generality.jl  NEW  gate 7
test/literal/pgm/generate_golden.py     NEW  regenerate golden from PGM
test/literal/pgm/golden.json            NEW  golden data (diffable)
test/literal/pgm/pgm_cases_data.jl      NEW  generated PGM_CASES block
test/literal/pgm/README.md              NEW
test/runtests.jl                       EDIT  register gates 6 & 7
examples/literal_pmdse_pgm_tutorial.jl  NEW  runnable verbose tutorial (entry point)
docs/src/literal_pmdse_tutorial.md      NEW  narrated tutorial w/ real matrices
docs/src/literal_pmdse.md              EDIT  document `method`
docs/src/Literal_PMDSE_PLAN.md         EDIT  document PGM options + gates 6/7
docs/make.jl                           EDIT  register tutorial page
CHANGELOG.md                           EDIT
```

The pre-existing Literal PMDSE design/usage docs remain the reference:
`docs/src/Literal_PMDSE_PLAN.md` (design + validation gates),
`docs/src/literal_pmdse.md` (API), `docs/src/literal_pmdse_observability.md`,
`examples/literal_pmdse_example.jl` (the legacy `:wls/:wlav/:mle` example).

---

## 8. PGM benchmark cases (for quick reference)

From `power-grid-model/tests/data/state_estimation/`:

| case | nodes | methods | exercises |
|---|---|---|---|
| `single-node-source-sym-voltage-sensor` | 1 | il, nr | trivial phasor pin |
| `1os2msr` | 3 | il, nr | redundant set, branch from/to power, source/load injection, a disabled (σ=1e15) sensor, Y-bus + to-side back-calc |
| `1os2msr-no-angle` | 3 | il, nr | magnitude-only → slack-angle gauge |
| `single-line-load-il` | 2 | il | inconsistent data (weighting matters), Kalman-combined duplicate load sensor |

PGM defaults: `error_tolerance=1e-8`, `max_iterations=20`,
`calculation_method=iterative_linear`.
