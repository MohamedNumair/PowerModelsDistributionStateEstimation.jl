################################################################################
#  Gate 6 : validate the Literal PMDSE PowerGridModel solve options              #
#  (`:iterative_linear`, `:newton_raphson`) against PowerGridModel itself.        #
#                                                                              #
#  Every case in `pgm_cases.jl` is a PGM state-estimation example whose node     #
#  voltages were computed by PGM (the golden `expected` field).  Here the same   #
#  network + sensors are fed to the literal estimator and the converged node     #
#  voltages (and line flows, hence the Y-bus and to-side branch back-calc) are   #
#  required to match PGM to tolerance — for *both* PGM methods, on the same       #
#  four-wire-general code path that also handles the explicit-neutral feeders.    #
################################################################################
using Test
import PowerModelsDistribution as _PMD
import PowerModelsDistributionStateEstimation as _PMDSE
import Distributions as _DST
import LinearAlgebra
include("pgm_cases.jl")

const _PGM_METHOD = Dict(:il => :iterative_linear, :nr => :newton_raphson)

"line (p_from,q_from,p_to,q_to) in SI from a literal solution, via the π-model"
function pgm_line_flows(case, res)
    sbase = PGM_SBASE; ω = 2π * PGM_FREQ
    vbase = Dict(n.id => n.u_rated for n in case.nodes)
    U(nid) = (b = res.solution["bus"][string(nid)]; (b["vr"][1] + im * b["vi"][1]))   # p.u.
    flows = Dict{Int,NamedTuple}()
    for l in case.lines
        zb = vbase[l.f]^2 / sbase
        ys = 1.0 / ((l.r1 + im * l.x1) / zb)
        b1 = ω * l.c1; g1 = l.tan1 * b1; ysh = ((g1 + im * b1) * zb) / 2
        Uf = U(l.f); Ut = U(l.t)
        Ifr = (ys + ysh) * Uf - ys * Ut
        Ito = (ys + ysh) * Ut - ys * Uf
        Sf = Uf * conj(Ifr) * sbase; St = Ut * conj(Ito) * sbase
        flows[l.id] = (p_from = real(Sf), q_from = imag(Sf), p_to = real(St), q_to = imag(St))
    end
    return flows
end

@testset "Literal PMDSE — PowerGridModel benchmark (gate 6)" begin
    for case in PGM_CASES
        @testset "$(case.name)" begin
            for m in case.methods
                math = build_pgm_math(case)
                build_pgm_meas!(math, case)
                res = _PMDSE.solve_mc_se_literal(math; estimator = :wls,
                        method = _PGM_METHOD[m], reference = :prop, maxiter = 100, tol = 1e-10)
                @test res.termination == :converged

                ex = case.expected[string(m)]
                maxe = 0.0
                for (nid, ev) in ex.nodes
                    Uest = pgm_result_voltage(case, res, nid)
                    Ugold = ev.u * exp(im * ev.u_angle)
                    maxe = max(maxe, abs(Uest - Ugold))
                    @test abs(Uest - Ugold) < 1e-3                  # SI volts (~1e-7 typical)
                    @test isapprox(abs(Uest), ev.u; rtol = 1e-6)
                end
                # Y-bus + to-side back-calculation: line flows must match PGM
                if !isempty(case.lines)
                    fl = pgm_line_flows(case, res)
                    for (lid, ev) in ex.lines
                        f = fl[lid]
                        @test isapprox(f.p_from, ev.p_from; rtol = 1e-5, atol = 1.0)
                        @test isapprox(f.q_from, ev.q_from; rtol = 1e-5, atol = 1.0)
                        @test isapprox(f.p_to,   ev.p_to;   rtol = 1e-5, atol = 1.0)
                        @test isapprox(f.q_to,   ev.q_to;   rtol = 1e-5, atol = 1.0)
                    end
                end
                @info "PGM benchmark" case = case.name method = _PGM_METHOD[m] max_U_err_V = maxe iters = res.iterations
            end
        end
    end

    # The two PGM methods must converge to the same WLS optimum (PGM ships both)
    @testset "iterative_linear == newton_raphson" begin
        for case in PGM_CASES
            (:il in case.methods && :nr in case.methods) || continue
            math = build_pgm_math(case); build_pgm_meas!(math, case)
            il = _PMDSE.solve_mc_se_literal(math; estimator = :wls, method = :iterative_linear,
                                            reference = :prop, maxiter = 100, tol = 1e-11)
            math2 = build_pgm_math(case); build_pgm_meas!(math2, case)
            nr = _PMDSE.solve_mc_se_literal(math2; estimator = :wls, method = :newton_raphson,
                                            reference = :prop, maxiter = 100, tol = 1e-11)
            d = 0.0
            for n in case.nodes
                d = max(d, abs(pgm_result_voltage(case, il, n.id) -
                               pgm_result_voltage(case, nr, n.id)))
            end
            @test d < 1e-2     # same state to well under a volt on a ~10 kV system
        end
    end
end
