################################################################################
#  Gate 3 : reproduce the observability theorem of main.tex.                      #
#  With only the neutral grounded (:prop) the global phase rotation              #
#  δx = [−vi; vr] is a null vector of H (power / |U| measurements are invariant   #
#  to a global rotation) ⇒ the gain is singular.  Adding one angle datum          #
#  (:sota) removes that null direction ⇒ the system becomes observable.           #
################################################################################
using Test
import PowerModelsDistribution as _PMD
import PowerModelsDistributionStateEstimation as _PMDSE
import Ipopt, ForwardDiff, LinearAlgebra
import Distributions as _DST
include("helpers.jl")

@testset "Literal PMDSE — angular reference / observability (gate 3)" begin
    math, pf = toy_pf()
    M = rotation_invariant_meas(math, pf)

    rot_resid = Dict{Symbol,Float64}()
    for ref in (:prop, :sota)
        lm = _PMDSE.LiteralModel(math; reference=ref)
        model = _PMDSE.build_se_model(lm, with_meas(math, M))
        xf = pf_xfree(lm, pf)
        H = ForwardDiff.jacobian(xx -> _PMDSE.predict(model, xx), xf)
        d = rotation_dir(lm, _PMDSE.expand_state(lm, xf))
        rot_resid[ref] = LinearAlgebra.norm(H * d) / (LinearAlgebra.norm(H) * LinearAlgebra.norm(d))
    end

    # without the angle datum the rotation is (numerically) unobservable …
    @test rot_resid[:prop] < 1e-8
    # … and with it the rotation becomes observable
    @test rot_resid[:sota] > 1e-3
end
