################################################################################
#  Gate 7 : the PowerGridModel solve options are GENERAL — they run unchanged on #
#  the four-wire, explicit-neutral feeder, not only on PGM's Kron-reduced         #
#  single-phase networks.                                                        #
#                                                                              #
#  The same `:iterative_linear` / `:newton_raphson` code path is fed the          #
#  canonical explicit-neutral IVREN measurement set (rectangular voltages +       #
#  per-conductor current injections, neutral kept explicit) on the 3-bus 4-wire   #
#  feeder.  It must (a) converge, (b) recover the IVREN power-flow state, and      #
#  (c) agree with the gate-4-validated `solve_wls`.  This exercises the neutral    #
#  handling (phase-to-neutral voltages, neutral injection `= −Σ phase currents`)  #
#  that PGM — being single-phase — never needs.                                   #
################################################################################
using Test
import PowerModelsDistribution as _PMD
import PowerModelsDistributionStateEstimation as _PMDSE
import Ipopt, ForwardDiff, LinearAlgebra
import Distributions as _DST
include("helpers.jl")

@testset "Literal PMDSE — four-wire EN generality of PGM solve options (gate 7)" begin
    math, pf = load_en_feeder("3bus_4wire.dss")
    canonical_meas!(math, pf; σ = 0.005)                 # noiseless EN current/voltage set
    math["se_settings"] = Dict{String,Any}("criterion" => "wls", "rescaler" => 1.0)
    refvals = ref_values_from_pf(math, pf)

    base = _PMDSE.solve_mc_se_literal(math; estimator = :wls, reference = :full_slack, ref_values = refvals)
    @test base.termination == :converged

    for method in (:iterative_linear, :newton_raphson)
        res = _PMDSE.solve_mc_se_literal(math; estimator = :wls, method = method,
                                         reference = :full_slack, ref_values = refvals,
                                         maxiter = 200, tol = 1e-9)
        @test res.termination == :converged
        # (b) recovers the explicit-neutral PF state, neutral included
        @test _PMDSE.accuracy_metrics(res, pf["solution"]; include_neutral = true).maxerr < 1e-4
        # (c) agrees with the gate-4-validated Gauss–Newton solve_wls
        agree = 0.0
        for (b, bsol) in base.solution["bus"], idx in eachindex(bsol["vr"])
            agree = max(agree, abs((res.solution["bus"][b]["vr"][idx] + im * res.solution["bus"][b]["vi"][idx]) -
                                   (bsol["vr"][idx] + im * bsol["vi"][idx])))
        end
        @test agree < 1e-4
        @info "EN generality" method = method maxerr_vs_PF =
              _PMDSE.accuracy_metrics(res, pf["solution"]).maxerr agree_vs_wls = agree iters = res.iterations
    end
end
