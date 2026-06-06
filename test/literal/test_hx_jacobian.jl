################################################################################
#  Gate 2 : the ForwardDiff Jacobian H = ∂h/∂x matches finite differences for    #
#  every supported measurement var-type.  Run on the well-scaled toy feeder so    #
#  that the steep magnitude/angle rows are not dominated by FD truncation; a      #
#  relative tolerance is used because FD truncation error scales with |∂h/∂x|.    #
################################################################################
using Test
import PowerModelsDistribution as _PMD
import PowerModelsDistributionStateEstimation as _PMDSE
import Ipopt, ForwardDiff, LinearAlgebra
import Distributions as _DST
include("helpers.jl")

@testset "Literal PMDSE — h(x) Jacobian vs finite differences (gate 2)" begin
    math = toy_math()
    lm = _PMDSE.LiteralModel(math; reference=:sota)
    model = _PMDSE.build_se_model(lm, with_meas(math, kitchen_sink_meas()))

    # measurement-type coverage check
    vars = Set(getfield(r, :var) for r in model.rowmeta if haskey(r, :var))
    for v in (:vr,:vi,:vm,:va,:vll,:cr,:ci,:cm,:ca,:p,:q,:inj_r,:inj_i)
        @test v in vars
    end

    rng_states = [0.05, -0.03, 0.07, 0.02, -0.06]
    maxrel = 0.0
    for s in rng_states
        x = _PMDSE.flat_start_free(lm) .+ s
        Jad = ForwardDiff.jacobian(xx -> _PMDSE.predict(model, xx), x)
        δ = 1e-6; Jfd = similar(Jad)
        for j in 1:length(x)
            xp = copy(x); xp[j]+=δ; xm = copy(x); xm[j]-=δ
            Jfd[:, j] = (_PMDSE.predict(model, xp) .- _PMDSE.predict(model, xm)) ./ (2δ)
        end
        maxrel = max(maxrel, maximum(abs.(Jad .- Jfd) ./ (1 .+ abs.(Jad))))
    end
    @test maxrel < 1e-6
end
