################################################################################
#  Gate 5 : benchmark the JuMP IVREN baseline against the literal estimators      #
#  (WLS & WLAV) over several noise seeds.  Headline metrics: estimation accuracy  #
#  vs ground truth (RMSE / max per-phase |U| incl. neutral) and compute           #
#  time / iterations.  Results are *compared*, not asserted equal.                #
################################################################################
using Test
import PowerModelsDistribution as _PMD
import PowerModelsDistributionStateEstimation as _PMDSE
import Ipopt, ForwardDiff, LinearAlgebra
import Distributions as _DST
import Statistics
using Printf
include("helpers.jl")

function ivren_maxerr(SE, pf)
    e = 0.0
    for (b, bsol) in SE["solution"]["bus"], idx in 1:length(bsol["vr"])
        e = max(e, abs((bsol["vr"][idx]+im*bsol["vi"][idx]) -
                       (pf["solution"]["bus"][b]["vr"][idx]+im*pf["solution"]["bus"][b]["vi"][idx])))
    end
    e
end

@testset "Literal PMDSE — IVREN vs literal benchmark (gate 5)" begin
    math0, pf = load_en_feeder("3bus_4wire.dss")
    refvals = ref_values_from_pf(math0, pf)
    σ = 0.01; seeds = [1, 2, 3, 11]
    slv = _PMDSE.optimizer_with_attributes(Ipopt.Optimizer, "tol"=>1e-9, "print_level"=>0)

    rows = NamedTuple[]
    for s in seeds
        math, _ = load_en_feeder("3bus_4wire.dss")
        path = joinpath(mktempdir(), "m$s.csv")
        _PMDSE.write_measurements!(_PMD.IVRENPowerModel, math, pf, path, σ=σ)
        _PMDSE.add_measurements!(math, path, actual_meas=false, seed=s)
        math["se_settings"] = Dict{String,Any}("criterion"=>"wls", "rescaler"=>1.0)

        SE = _PMDSE.solve_ivr_en_mc_se(math, slv)
        wls  = _PMDSE.solve_mc_se_literal(math; estimator=:wls,  reference=:full_slack, ref_values=refvals)
        wlav = _PMDSE.solve_mc_se_literal(math; estimator=:wlav, reference=:full_slack, ref_values=refvals)

        wls_err  = _PMDSE.accuracy_metrics(wls,  pf["solution"]).maxerr
        wlav_err = _PMDSE.accuracy_metrics(wlav, pf["solution"]).maxerr
        push!(rows, (seed=s,
            ivren=(ivren_maxerr(SE, pf), get(SE, "solve_time", NaN)),
            wls =(wls_err,  wls.solve_time,  wls.iterations),
            wlav=(wlav_err, wlav.solve_time, wlav.iterations)))

        @test wls.termination == :converged
        @test wlav.termination == :converged
        @test SE["termination_status"] ∈ [_PMDSE.LOCALLY_SOLVED, _PMDSE.ALMOST_LOCALLY_SOLVED]
        # literal accuracy is comparable to the JuMP baseline (not asserted equal)
        @test wls_err < 10σ
    end

    println("\n  IVREN vs Literal PMDSE — 3-bus 4-wire, σ=$(σ)")
    println("  seed |  IVREN max|U|  t[s]  |  WLS max|U|  t[s]  it |  WLAV max|U|  t[s]  it")
    for r in rows
        @printf("  %4d | %12.2e %5.2f | %10.2e %5.3f %3d | %10.2e %5.3f %3d\n",
                r.seed, r.ivren[1], r.ivren[2], r.wls[1], r.wls[2], r.wls[3],
                r.wlav[1], r.wlav[2], r.wlav[3])
    end
    avg(f) = Statistics.mean(f(r) for r in rows)
    @printf("  mean | %12.2e       | %10.2e             | %10.2e\n",
            avg(r->r.ivren[1]), avg(r->r.wls[1]), avg(r->r.wlav[1]))
end
