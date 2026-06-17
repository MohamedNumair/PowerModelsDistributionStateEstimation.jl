################################################################################
#  Gate 4 : on a noiseless, consistent measurement set with identical references, #
#  the literal WLS converges to the same state as `solve_ivr_en_mc_se`            #
#  (IVRENPowerModel).  This pins down every sign / convention.                    #
################################################################################
using Test
import PowerModelsDistribution as _PMD
import PowerModelsDistributionStateEstimation as _PMDSE
import Ipopt, ForwardDiff, LinearAlgebra
import Distributions as _DST
include("helpers.jl")

@testset "Literal PMDSE — WLS vs IVREN (gate 4)" begin
    math, pf = load_en_feeder("3bus_4wire.dss")
    canonical_meas!(math, pf; σ=0.005)          # noiseless: values == PF
    math["se_settings"] = Dict{String,Any}("criterion"=>"wls", "rescaler"=>1.0)
    refvals = ref_values_from_pf(math, pf)

    @testset "noiseless literal WLS reproduces the PF state" begin
        for ref in (:full_slack, :sota)
            rv = ref == :full_slack ? refvals : nothing
            res = _PMDSE.solve_mc_se_literal(math; estimator=:wls, reference=ref, ref_values=rv)
            @test res.termination == :converged
            @test _PMDSE.accuracy_metrics(res, pf["solution"]).maxerr < 1e-6
        end
    end

    @testset "Gaussian MLE reduces to WLS" begin
        wls = _PMDSE.solve_mc_se_literal(math; estimator=:wls, reference=:sota)
        mle = _PMDSE.solve_mc_se_literal(math; estimator=:mle, reference=:sota)
        @test mle.termination == :converged
        @test maximum(abs.(mle.x_free .- wls.x_free)) < 1e-8
    end

    @testset "literal WLS matches the JuMP IVREN estimator" begin
        slv = _PMDSE.optimizer_with_attributes(Ipopt.Optimizer, "tol"=>1e-10, "print_level"=>0)
        SE = _PMDSE.solve_ivr_en_mc_se(math, slv)
        res = _PMDSE.solve_mc_se_literal(math; estimator=:wls, reference=:full_slack, ref_values=refvals)
        maxdiff = 0.0
        for (b, bsol) in SE["solution"]["bus"]
            for idx in 1:length(bsol["vr"])
                maxdiff = max(maxdiff, abs((res.solution["bus"][b]["vr"][idx] + im*res.solution["bus"][b]["vi"][idx]) -
                                           (bsol["vr"][idx] + im*bsol["vi"][idx])))
            end
        end
        @test maxdiff < 1e-4        # both recover the same physical state
    end

    @testset "noisy literal WLS stays close to ground truth" begin
        mathn, _ = load_en_feeder("3bus_4wire.dss")
        path = joinpath(mktempdir(), "mn.csv")
        _PMDSE.write_measurements!(_PMD.IVRENPowerModel, mathn, pf, path, σ=0.01)
        _PMDSE.add_measurements!(mathn, path, actual_meas=false, seed=7)   # sampled noise
        mathn["se_settings"] = Dict{String,Any}("criterion"=>"wls", "rescaler"=>1.0)
        res = _PMDSE.solve_mc_se_literal(mathn; estimator=:wls, reference=:full_slack,
                                         ref_values=ref_values_from_pf(mathn, pf))
        @test res.termination == :converged
        @test _PMDSE.accuracy_metrics(res, pf["solution"]).maxerr < 0.05
    end
end
