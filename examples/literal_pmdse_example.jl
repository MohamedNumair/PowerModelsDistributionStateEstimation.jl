################################################################################
#  Literal PMDSE — worked example                                               #
#                                                                              #
#  Demonstrates the explicit, JuMP-free explicit-neutral state estimator        #
#  `solve_mc_se_literal` with every estimator (:wls/:wlav/:mle) and reference    #
#  scheme (:sota/:full_slack/:prop), result inspection, accuracy metrics and     #
#  the lower-level API.                                                          #
#                                                                              #
#  Run with a project that has Ipopt + PowerModelsDistribution(StateEstimation):#
#      julia --project examples/literal_pmdse_example.jl                         #
################################################################################
import Ipopt
import ForwardDiff, LinearAlgebra
import PowerModelsDistribution as _PMD
import PowerModelsDistributionStateEstimation as _PMDSE

ipopt = _PMDSE.optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0, "tol"=>1e-10)

# ---------------------------------------------------------------------------- #
# 1. Ground truth: parse a 4-wire explicit-neutral feeder and run a power flow  #
# ---------------------------------------------------------------------------- #
eng = _PMD.parse_file(joinpath(_PMDSE.BASE_DIR, "test", "data",
                               "three-bus-en-models", "3bus_4wire.dss"))
_PMD.transform_loops!(eng)
_PMD.remove_all_bounds!(eng)
math = _PMD.transform_data_model(eng, kron_reduce=false, phase_project=false)
_PMD.add_start_vrvi!(math)
pf = _PMD.solve_mc_opf(math, _PMD.IVRENPowerModel, ipopt)

# ---------------------------------------------------------------------------- #
# 2. Synthesize the canonical IVREN measurement set (vr/vi + crd/cid + crg/cig) #
#    `actual_meas=true` keeps the exact PF values (noiseless); use a seed for    #
#    sampled noise (see step 6).                                                 #
# ---------------------------------------------------------------------------- #
msr = joinpath(mktempdir(), "msr.csv")
_PMDSE.write_measurements!(_PMD.IVRENPowerModel, math, pf, msr, σ=0.005)
_PMDSE.add_measurements!(math, msr, actual_meas=true)
math["se_settings"] = Dict{String,Any}("rescaler" => 1.0)   # weights = 1/(rescaler·σ)²

# ---------------------------------------------------------------------------- #
# 3. Run every estimator with the default SOTA reference                        #
# ---------------------------------------------------------------------------- #
println("="^78, "\n3.  estimators (reference = :sota)\n", "="^78)
for est in (:wls, :wlav, :mle)
    res = _PMDSE.solve_mc_se_literal(math; estimator=est, reference=:sota,
                                     maxiter=50, tol=1e-9, verbose=false)
    m = _PMDSE.accuracy_metrics(res, pf["solution"])
    println(rpad(uppercase(string(est)), 5),
            " term=", res.termination, " iters=", res.iterations,
            "  rmse|U|=", round(m.rmse, sigdigits=3),
            "  max|U|=",  round(m.maxerr, sigdigits=3),
            "  t=", round(res.solve_time, sigdigits=3), "s")
end

# ---------------------------------------------------------------------------- #
# 4. Reference schemes                                                          #
#    :sota       — ground neutral + one angle datum (estimates the unbalanced RB)#
#    :full_slack — fix the whole reference-bus phasor (here, to the true values) #
#    :prop       — neutral only -> rank-deficient (unobservable); shown for study#
# ---------------------------------------------------------------------------- #
println("\n", "="^78, "\n4.  reference schemes (estimator = :wls)\n", "="^78)
rb   = _PMDSE.literal_ref_bus(math)
bsol = pf["solution"]["bus"]["$rb"]; terms = math["bus"]["$rb"]["terminals"]
refvals = (Dict(t => bsol["vr"][i] for (i, t) in enumerate(terms)),
           Dict(t => bsol["vi"][i] for (i, t) in enumerate(terms)))

for (ref, rv) in ((:sota, nothing), (:full_slack, refvals))
    res = _PMDSE.solve_mc_se_literal(math; estimator=:wls, reference=ref, ref_values=rv)
    m = _PMDSE.accuracy_metrics(res, pf["solution"])
    println(rpad(string(ref), 11), " term=", res.termination,
            "  max|U vs PF|=", round(m.maxerr, sigdigits=3),
            "  cond(G)=", round(res.gain_cond, sigdigits=3))
end

# ---------------------------------------------------------------------------- #
# 5. Inspect a result object                                                    #
# ---------------------------------------------------------------------------- #
println("\n", "="^78, "\n5.  result inspection\n", "="^78)
res = _PMDSE.solve_mc_se_literal(math; estimator=:wls, reference=:full_slack, ref_values=refvals)
println("termination = ", res.termination, "   iterations = ", res.iterations)
println("objective   = ", round(res.objective, sigdigits=3))
println("gain rank   = ", res.gain_rank)
println("loadbus |U| (per terminal incl. neutral) = ", round.(res.solution["bus"]["3"]["vm"], digits=5))
println("per-node |ΔU| vs PF (first 6) = ", round.(_PMDSE.voltage_errors(res, pf["solution"])[1:6], sigdigits=2))

# ---------------------------------------------------------------------------- #
# 6. Robustness: a single gross error — WLAV vs WLS                             #
# ---------------------------------------------------------------------------- #
println("\n", "="^78, "\n6.  one gross error: WLAV is robust, WLS is not\n", "="^78)
mathb = deepcopy(math)
mkey  = first(k for (k, mm) in mathb["meas"] if mm["var"] == :vr)   # corrupt a vr meas
import Distributions as _DST
mathb["meas"][mkey]["dst"][1] = _DST.Normal(_DST.mean(mathb["meas"][mkey]["dst"][1]) + 0.3, 0.005)
for est in (:wls, :wlav)
    res = _PMDSE.solve_mc_se_literal(mathb; estimator=est, reference=:full_slack, ref_values=refvals)
    println(rpad(uppercase(string(est)), 5), " max|U vs PF| = ",
            round(_PMDSE.accuracy_metrics(res, pf["solution"]).maxerr, sigdigits=3))
end

# ---------------------------------------------------------------------------- #
# 7. Lower-level API: build the model once, reuse it, get the Jacobian          #
# ---------------------------------------------------------------------------- #
println("\n", "="^78, "\n7.  lower-level API\n", "="^78)
lm    = _PMDSE.LiteralModel(math; reference=:full_slack, ref_values=refvals)
model = _PMDSE.build_se_model(lm, math; rescaler=1.0)
res   = _PMDSE.solve_wls(model)
H     = ForwardDiff.jacobian(x -> _PMDSE.predict(model, x), res.x_free)
println("residual rows = ", length(model.z), "   free state dim = ", length(lm.free_idx))
println("size(H) = ", size(H), "   cond(HᵀWH) = ", round(res.gain_cond, sigdigits=3))
println("\nDONE.")
