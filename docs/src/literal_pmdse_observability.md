# How Literal PMDSE resolves the angular & zero-sequence reference problem

This page connects the [Literal PMDSE — explicit matrix-based estimator](@ref)
estimator to the theory of the report *"Rigorous Mathematical Analysis of the
Angular and Zero-Sequence Reference Problem in 4-Wire Unbalanced IVR Power Flow"*
(`main.tex`, TSK-871). Literal PMDSE is the explicit, matrix-based estimator that
**operationalizes** that report's conclusion and turns its central theorem into a
runnable regression test.

## The problem (recap of `main.tex`)

In a 4-wire explicit-neutral IVR model the state is the rectangular nodal voltage
``\mathbf{U}\in\mathbb{C}^{4n}`` and the measurement set of interest consists of
power / current-magnitude / voltage-magnitude quantities plus the explicit
grounding of the reference-bus neutral ``U_{s,n}=0``.

**Theorem 1 (Rotational Invariance).** *Grounding only the neutral leaves the
Jacobian singular, rank-deficient by exactly one.* A global phase rotation
``\mathbf{U}\mapsto \mathbf{U}\,e^{j\alpha}`` (i) preserves the neutral constraint
(``0\cdot e^{j\alpha}=0``) and (ii) leaves every complex power injection
invariant (``S_{k,p}=U_{k,p}\,(\sum Y\,U)^\ast`` ⇒ the ``e^{j\alpha}`` and
``e^{-j\alpha}`` cancel). Hence there is a continuous locus of solutions and the
measurement Jacobian ``H`` has the null vector

```math
\frac{\partial \mathbf{U}(\alpha)}{\partial\alpha}\Big|_{\alpha=0}
   = j\,\mathbf{U}^\ast
   \;\;\Longleftrightarrow\;\;
   \delta\mathbf{x}_{\mathrm{rot}}=\begin{bmatrix}-\mathbf{U}^{\mathrm i}\\[2pt]\;\;\mathbf{U}^{\mathrm r}\end{bmatrix}.
```

Because ``H`` is not full column rank, the gain ``G=H^\top W H`` is **singular**
and the system is **unobservable**.

**Resolution.** Grounding the neutral fixes only the *translational / zero-sequence*
datum. The *rotational / positive-sequence* datum must be fixed separately by one
phase-angle reference ``U^{\mathrm i}_{s,a}=0``. The SOTA estimator therefore
**combines** both:

1. ground the reference-bus neutral — ``U^{\mathrm r}_{s,n}=U^{\mathrm i}_{s,n}=0``;
2. fix a single angle datum — ``U^{\mathrm i}_{s,a}=0``.

## How Literal PMDSE implements it

Literal PMDSE exposes the two constraints above (and the failing proposition) as
the configurable `reference` keyword of `solve_mc_se_literal` / `LiteralModel`.
The reference partition removes the fixed components as *columns* of ``H`` (they
become known parameters), so the remaining gain is full-rank **without** assuming
a balanced reference bus:

| `main.tex` case | Literal PMDSE `reference` | what is fixed | rank of ``H`` |
|-----------------|---------------------------|---------------|---------------|
| Proposition (neutral only) | `:prop`  | ``U^{\mathrm r}_{s,n}=U^{\mathrm i}_{s,n}=0`` | **deficient by 1** |
| Ultimate solution (SOTA) | `:sota`  | neutral **and** ``U^{\mathrm i}_{s,a}=0`` | full |
| Conventional fixed slack | `:full_slack` | the whole reference-bus phasor | full (forces balanced RB) |

`:sota` is the default, and — exactly as the report prescribes — it leaves the
reference-bus phase ``b``/``c`` magnitudes and angles **free state variables**, so
the unbalanced reference bus and the neutral-to-earth voltage (NEV) are estimated
rather than assumed.

## The theorem as a regression test

The report's negative result is encoded directly in `test/literal/test_references.jl`.
Evaluate the Jacobian ``H`` at the true state and probe it along the analytic
rotation null vector ``\delta\mathbf{x}_{\mathrm{rot}}=[-\mathbf{U}^{\mathrm i};\,
\mathbf{U}^{\mathrm r}]`` (restricted to the free coordinates):

```julia
import PowerModelsDistributionStateEstimation as _PMDSE
import ForwardDiff, LinearAlgebra

# rotation-invariant measurements (|U|, P, Q) on the toy 3-bus 4-wire feeder
math, pf = toy_pf()                 # see test/literal/helpers.jl
M        = rotation_invariant_meas(math, pf)

for reference in (:prop, :sota)
    lm    = _PMDSE.LiteralModel(math; reference=reference)
    model = _PMDSE.build_se_model(lm, with_meas(math, M))
    xf    = pf_xfree(lm, pf)
    H     = ForwardDiff.jacobian(x -> _PMDSE.predict(model, x), xf)
    d     = rotation_dir(lm, _PMDSE.expand_state(lm, xf))      # δx_rot, free coords
    @show reference, LinearAlgebra.norm(H*d) / (LinearAlgebra.norm(H)*LinearAlgebra.norm(d))
end
```

which yields

```
(:prop, 2.56e-17)     # H·δx_rot ≈ 0  →  rotation is UNOBSERVABLE (Theorem 1)
(:sota, 5.12e-02)     # adding the angle datum makes the rotation observable
```

The rotation is a numerical null vector of ``H`` under `:prop` and is removed
under `:sota` — a direct, quantitative reproduction of Theorem 1 and its
resolution.

## The toy 3-bus 4-wire example

`main.tex` §6 analyses a 3-bus, 4-wire radial network
(``\mathbf{Z}_{abc}=\mathrm{diag}(0.01+j0.02)\,\Omega``,
``\mathbf{Z}_n=0.05+j0.01\,\Omega``, 100 kVA / 400 V) and reports
``\mathrm{rank}(\mathbf J)=23`` (singular) with neutral-only grounding,
restored to ``\mathrm{rank}(\mathbf J)=24`` once ``U^{\mathrm i}_{1,a}=0`` is
added. The same feeder is reproduced in `test/literal/helpers.jl` (`toy_math`)
and used by the observability test above, so the report's worked example and the
estimator share one network.

## Validation against the JuMP IVREN estimator

Beyond observability, Literal PMDSE is validated to **agree** with the
optimization-based explicit-neutral estimator (`solve_ivr_en_mc_se` /
`IVRENPowerModel`). On a noiseless, consistent measurement set with identical
references, the literal WLS reproduces the IVREN / power-flow state to machine
precision (`max|ΔU| ~ 1e-16` for both `:full_slack` and `:sota`), while
converging in ~2 Gauss–Newton iterations and running roughly two orders of
magnitude faster than the Ipopt solve (see `test/literal/test_benchmark.jl`).

## Summary

| concern (`main.tex`) | mechanism | Literal PMDSE |
|----------------------|-----------|---------------|
| zero-sequence / translational datum | ground reference-bus neutral | always (grounded terminals fixed to 0) |
| positive-sequence / rotational datum | fix one phase angle | `:sota` adds ``U^{\mathrm i}_{s,a}=0`` |
| unbalanced reference bus (NEV) | leave ``b,c`` free | `:sota` estimates them |
| the negative result (singular gain) | rotational invariance | reproduced by `:prop` (rank-deficient by 1) |

See also: [Angular Reference Models](@ref) and
[Explicit Neutral Models for DSSE](@ref).
