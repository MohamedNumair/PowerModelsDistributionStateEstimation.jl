################################################################################
#  Gate 1 : the assembled Ybus reproduces the network physics.                   #
#  PMD's `calc_admittance_matrix` does not support the 4-wire explicit-neutral   #
#  (kron_reduce=false) model, so we validate against PMD's own power-flow:        #
#  `Ybus·U` must equal the component current injection (Σgen − Σload) at every    #
#  injection node, which is exactly the KCL that `IVRENPowerModel` enforces.      #
################################################################################
using Test
import PowerModelsDistribution as _PMD
import PowerModelsDistributionStateEstimation as _PMDSE
import Ipopt, ForwardDiff, LinearAlgebra
import Distributions as _DST
include("helpers.jl")

@testset "Literal PMDSE — Ybus assembly (gate 1)" begin
    math, pf = load_en_feeder("3bus_4wire.dss")
    lm = _PMDSE.LiteralModel(math; reference=:sota)

    @test LinearAlgebra.norm(lm.G .- transpose(lm.G), Inf) < 1e-8   # Ybus symmetric
    @test LinearAlgebra.norm(lm.B .- transpose(lm.B), Inf) < 1e-8

    U = pf_voltages(lm, pf); vr = real.(U); vi = imag.(U)
    Ir, Ii = _PMDSE.nodal_injection(lm, vr, vi)
    Icalc = Ir .+ im .* Ii
    Icmp  = pf_component_injection(lm, pf)

    maxerr = maximum(abs(Icalc[k] - Icmp[k]) for k in lm.inj_nodes)
    @test maxerr < 1e-6          # Ybus·U == component injection (Σgen − Σload)

    # toy network: well-scaled Ybus must also be symmetric and finite
    lmt = _PMDSE.LiteralModel(toy_math(); reference=:sota)
    @test all(isfinite, lmt.G) && all(isfinite, lmt.B)
    @test LinearAlgebra.norm(lmt.G .- transpose(lmt.G), Inf) < 1e-10
end
