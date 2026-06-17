################################################################################
#  Copyright 2020, Marta Vanin, Tom Van Acker                                  #
################################################################################
# PowerModelsDistributionStateEstimation.jl                                    #
# An extention package of PowerModels(Distribution).jl for Static Power System #
# State Estimation.                                                            #
################################################################################

import PowerModelsDistributionStateEstimation as _PMDSE

# import pkgs
import Distributions as _DST
import HDF5
import Ipopt
import Polynomials as _Poly
import PowerModels
import PowerModelsDistribution as _PMD
import Statistics
using Test
using SafeTestsets

#network and feeder from ENWL for tests
ntw, fdr = 4, 2

season     = "summer"
time_step  = 144
elm        = ["load", "pv"]
pfs        = [0.95, 0.90]
rm_transfo = true
rd_lines   = true

# set solvers
ipopt_solver = _PMDSE.optimizer_with_attributes(Ipopt.Optimizer,"max_cpu_time" => 300.0,
                                                         "obj_scaling_factor" => 1e3,
                                                         "tol" => 1e-9,
                                                         "print_level" => 0, 
                                                         "mu_strategy" => "adaptive")

@testset "PowerModelsDistributionStateEstimation" begin
    include("test_utils.jl")
    include("bad_data.jl")
    include("distributions.jl")
    include("estimation_criteria.jl")
    include("ivren.jl")
    include("mixed_measurements.jl")
    include("non_exact_forms.jl")
    include("power_flow.jl")
    include("pseudo_measurements.jl")
    include("reference_angles.jl")
    include("single_conductor_branches.jl")
    include("utils_and_start_val.jl")
    include("with_errors.jl")
end

# Literal PMDSE: explicit matrix-based explicit-neutral estimator (JuMP-free)
@safetestset "Literal PMDSE — Ybus assembly"   begin include("literal/test_ybus.jl")        end
@safetestset "Literal PMDSE — h(x) Jacobian"   begin include("literal/test_hx_jacobian.jl")  end
@safetestset "Literal PMDSE — references"       begin include("literal/test_references.jl")   end
@safetestset "Literal PMDSE — WLS vs IVREN"     begin include("literal/test_wls_vs_ivren.jl") end
@safetestset "Literal PMDSE — benchmark"        begin include("literal/test_benchmark.jl")    end
@safetestset "Literal PMDSE — PGM benchmark"     begin include("literal/test_pgm_benchmark.jl")     end
@safetestset "Literal PMDSE — PGM EN generality" begin include("literal/test_pgm_en_generality.jl") end

ambiguities = Test.detect_ambiguities(_PMDSE);
if !isempty(ambiguities)
    println("ambiguities detected: $ambiguities")
end