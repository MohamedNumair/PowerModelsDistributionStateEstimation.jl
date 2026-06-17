################################################################################
#  Literal PMDSE — PowerGridModel tutorial (verbose entry point)                 #
#                                                                              #
#  A heavily-commented, matrix-by-matrix walk through the two PowerGridModel     #
#  (PGM) state-estimation solve options reproduced in Literal PMDSE:            #
#      :iterative_linear   and   :newton_raphson                                #
#                                                                              #
#  We build everything from scratch so you can see every intermediate object:   #
#  the per-unit data dictionary, the nodal admittance matrix Y_bus, the         #
#  measurement model (one "atom" per scalar measurement), the linear            #
#  measurement matrix A and weights W used by iterative-linear, the per-        #
#  iteration voltage updates, the Newton-Raphson Jacobian H and gain HᵀWH, and  #
#  finally a comparison against PowerGridModel's own published results.          #
#                                                                              #
#  Part A : a 3-bus single-phase PGM example  (power-grid-model 1os2msr case)    #
#  Part B : the SAME code on a four-wire, explicit-neutral network              #
#  Part C : the other reference schemes and estimators                          #
#                                                                              #
#  Run (needs the package instantiated):                                        #
#      julia --project examples/literal_pmdse_pgm_tutorial.jl                    #
################################################################################
import PowerModelsDistributionStateEstimation as _PMDSE
import PowerModelsDistribution as _PMD
import Distributions as _DST
import LinearAlgebra
const PMDSE = _PMDSE

hr(t) = println("\n", "="^78, "\n  ", t, "\n", "="^78)
cfmt(z) = string(round(real(z); digits=2), imag(z) ≥ 0 ? "+" : "", round(imag(z); digits=2), "im")
function showmat(name, M; cplx = eltype(M) <: Complex)
    println("\n", name, "  (", size(M, 1), "×", size(M, 2), ")")
    for i in 1:size(M, 1)
        print("   ")
        for j in 1:size(M, 2)
            print(lpad(cplx ? cfmt(M[i, j]) : string(round(M[i, j]; digits=4)), cplx ? 16 : 10), " ")
        end
        println()
    end
end

############################################################################ A
hr("PART A — 3-bus single-phase PGM example (the `1os2msr` validation case)")

# --- A.1  the network, exactly as in power-grid-model/tests/data/state_estimation/1os2msr
#     3 nodes @ 10.5 kV, 2 lines, a source at node 1, loads at nodes 2 & 3.
#     PGM works in SI; Literal PMDSE (like PMD) works in per-unit.  We pick the
#     SAME bases PGM uses internally: V_base = u_rated, S_base = base_power_3p = 1e6.
u_rated = 10.5e3
Sbase   = 1.0e6
Zbase   = u_rated^2 / Sbase                  # ≈ 110.25 Ω
ω       = 2π * 50.0
println("\nper-unit bases:  V_base = $(u_rated) V   S_base = $(Sbase) VA   Z_base = $(round(Zbase;digits=3)) Ω")

# PGM line parameters (Ω, F): r1, x1, c1, tan1 — π-model, shunt split half each end.
lines = [(id=4, f=1, t=2, r1=0.11, x1=0.12, c1=4.1380285203892784e-5, tan1=0.1076923076923077),
         (id=5, f=2, t=3, r1=0.15, x1=0.16, c1=5.411268065124442e-5, tan1=0.10588235294117646)]

# --- A.2  build the PMD *mathematical* data dictionary (per-unit, 1 conductor/bus)
function build_branch(l)
    b1 = ω * l.c1; g1 = l.tan1 * b1               # shunt: b = ωC,  g = tan·b
    gsh = (g1 * Zbase) / 2; bsh = (b1 * Zbase) / 2 # half to each π end, in p.u.
    Dict{String,Any}("index"=>l.id, "f_bus"=>l.f, "t_bus"=>l.t,
        "f_connections"=>[1], "t_connections"=>[1],
        "br_r"=>reshape([l.r1/Zbase],1,1), "br_x"=>reshape([l.x1/Zbase],1,1),
        "g_fr"=>reshape([gsh],1,1), "b_fr"=>reshape([bsh],1,1),
        "g_to"=>reshape([gsh],1,1), "b_to"=>reshape([bsh],1,1), "br_status"=>1)
end
bus(i, bt) = Dict{String,Any}("index"=>i, "bus_i"=>i, "bus_type"=>bt, "terminals"=>[1],
        "grounded"=>Bool[false], "vbase"=>u_rated, "vmin"=>[0.0], "vmax"=>[Inf],
        "vr_start"=>[1.0], "vi_start"=>[0.0])
math = Dict{String,Any}(
    "bus"    => Dict("1"=>bus(1,3), "2"=>bus(2,1), "3"=>bus(3,1)),         # node 1 = reference
    "branch" => Dict(string(l.id)=>build_branch(l) for l in lines),
    "gen"    => Dict("6"=>Dict{String,Any}("index"=>6,"gen_bus"=>1,"connections"=>[1],"configuration"=>_PMD.WYE)),
    "load"   => Dict("7"=>Dict{String,Any}("index"=>7,"load_bus"=>2,"connections"=>[1],"configuration"=>_PMD.WYE,"pd"=>[0.0],"qd"=>[0.0],"model"=>_PMD.POWER,"status"=>1),
                     "8"=>Dict{String,Any}("index"=>8,"load_bus"=>3,"connections"=>[1],"configuration"=>_PMD.WYE,"pd"=>[0.0],"qd"=>[0.0],"model"=>_PMD.POWER,"status"=>1)),
    "shunt"  => Dict{String,Any}(), "settings"=>Dict{String,Any}("sbase"=>Sbase))

# --- A.3  the nodal admittance matrix Y_bus (the heart of the network model)
lm = PMDSE.LiteralModel(math; reference = :prop)   # :prop = fix only grounded terminals
println("\nnode order (bus, terminal): ", lm.nodes)
Ybus = complex.(lm.G, lm.B)
showmat("Y_bus  [p.u.]  (I = Y_bus · U)", Ybus)

# --- A.4  the measurements, in real units (volts / watts / vars), as PGM gives them
#     voltage sensors carry a magnitude AND an angle (a phasor) -> rectangular vr/vi;
#     power sensors sit on a branch terminal (from/to) or an appliance (source/load).
V(node,u,θ,σ) = (cmp=:bus, id=node, vars=[(:vr, u*cos(θ)), (:vi, u*sin(θ))], σ=σ)
vsens = [V(1, 10751.072595758282, -0.013054638926306409, 105.0),
         V(2, 10752.698591183394, -0.017637349459726520, 105.0),
         V(3, 10748.320749959701, -0.020182330759474040, 100.0)]
# (kind, object, side/appliance, P, Q, σ)
psens = [(:branch, 4, :from, 2412359.2976399013, -3024028.886598367, 37916.0),
         (:branch, 4, :to,  -2000000.0,            1000000.0,          1e15),   # disabled (σ huge)
         (:branch, 5, :from, 1230426.390004009,  -1742195.1033582848, 20878.0),
         (:branch, 5, :to,  -1019999.9999999485,  -219999.99999999927, 10435.0),
         (:source, 6, :_,    2412359.297639887,  -3024028.8865982923, 38009.0),
         (:load,   7, :_,    1010000.0,            210000.0,           10316.0),
         (:load,   8, :_,    1020000.0,            220000.0,           10435.0)]

#     write data["meas"] in the package vocabulary (per-unit), exactly as
#     `add_measurements!` would produce it.
N(μ,σ) = _DST.Normal(μ, σ)
meas = Dict{String,Any}(); mid = Ref(0)
addmeas!(d) = (mid[] += 1; meas[string(mid[])] = d)
for s in vsens, (var, val) in s.vars
    addmeas!(Dict{String,Any}("cmp"=>:bus, "cmp_id"=>s.id, "var"=>var, "dst"=>[N(val/u_rated, s.σ/u_rated)]))
end
for (kind, obj, side, P, Q, σ) in psens
    if kind == :branch
        addmeas!(Dict{String,Any}("cmp"=>:branch,"cmp_id"=>obj,"var"=>:p,"side"=>side,"dst"=>[N(P/Sbase, σ/Sbase)]))
        addmeas!(Dict{String,Any}("cmp"=>:branch,"cmp_id"=>obj,"var"=>:q,"side"=>side,"dst"=>[N(Q/Sbase, σ/Sbase)]))
    else
        cmp, pv, qv = kind == :source ? (:gen, :pg, :qg) : (:load, :pd, :qd)
        addmeas!(Dict{String,Any}("cmp"=>cmp,"cmp_id"=>obj,"var"=>pv,"dst"=>[N(P/Sbase, σ/Sbase)]))
        addmeas!(Dict{String,Any}("cmp"=>cmp,"cmp_id"=>obj,"var"=>qv,"dst"=>[N(Q/Sbase, σ/Sbase)]))
    end
end
math["meas"] = meas
println("\nbuilt ", length(meas), " scalar measurement entries (var=:vr/:vi/:p/:q/:pd/:qd/:pg/:qg)")

# --- A.5  the measurement model: each measurement becomes an "atom"
#     ΔU(x) = cu·U  (a phase-to-neutral voltage)   I(x) = ci·U  (a branch/injection current)
atoms = PMDSE.build_se_atoms(lm, math; rescaler = 1.0)
println("\nmeasurement atoms (", length(atoms), "):")
for a in atoms
    tag = a.kind in (:vre,:vim) ? "z=$(round(a.z1;digits=4))  σ=$(round(a.sigma;digits=4))" :
          a.kind == :power      ? "P=$(round(a.z1;digits=3)) Q=$(round(a.z2;digits=3)) σ=$(round(a.sigp;digits=4))" :
          a.kind == :cinj       ? "Ir=$(round(a.z1;digits=3)) Ii=$(round(a.z2;digits=3))" : ""
    println("   ", rpad(string(a.kind),6), "  ", tag)
end

# --- A.6  iterative-linear: reconstruct the linear system A·x = b it solves
#     (this is exactly what `solve_se_il` builds; we rebuild it here to see it).
function il_matrix(atoms, n)
    rows = Vector{Float64}[]
    for a in atoms
        if     a.kind == :vre  ; push!(rows, vcat(a.cur, -a.cui))
        elseif a.kind == :vim  ; push!(rows, vcat(a.cui,  a.cur))
        elseif a.kind == :vmag ; push!(rows, vcat(a.cur, -a.cui)); push!(rows, vcat(a.cui, a.cur))
        else                     push!(rows, vcat(a.cir, -a.cii)); push!(rows, vcat(a.cii, a.cir))
        end
    end
    return permutedims(reduce(hcat, rows))
end
A = il_matrix(atoms, lm.n)
showmat("iterative-linear measurement matrix A  (rows = real/imag of every measurement; cols = [vr₁ vr₂ vr₃ vi₁ vi₂ vi₃])", A[1:min(8,end), :])
println("   … (", size(A,1), " rows total)")

hr("A.7 — run iterative_linear (watch max|ΔU| shrink each iteration)")
res_il = PMDSE.solve_mc_se_literal(math; estimator = :wls, method = :iterative_linear,
                                   reference = :prop, maxiter = 50, tol = 1e-10, verbose = true)
println("\nconverged: ", res_il.termination, " in ", res_il.iterations, " iterations,",
        "  weighted SSR = ", round(res_il.objective; sigdigits = 4))

hr("A.8 — run newton_raphson and inspect the Gauss-Newton gain matrix")
res_nr = PMDSE.solve_mc_se_literal(math; estimator = :wls, method = :newton_raphson,
                                   reference = :prop, maxiter = 50, tol = 1e-10)
#   reconstruct H and the gain G = HᵀWH at the solution (the normal-equation matrix)
free, xfix, _ = PMDSE._gauge(lm, atoms); xf = res_nr.x_full[free]
H = _PMDSE.ForwardDiff.jacobian(xx -> PMDSE._nr_predict(atoms, lm.n, xfix, free, xx), xf)
z, w = PMDSE._nr_zw(atoms); G = transpose(H) * (w .* H)
println("Newton-Raphson:  ", res_nr.termination, " in ", res_nr.iterations, " iteration(s)")
showmat("gain  G = HᵀWH   (note the ~1e10 entries: the per-unit line impedances make G ill-conditioned;\n   the QR/orthogonal fallback in solve_wls/solve_se_nr handles it)", G)

hr("A.9 — compare the estimate to PowerGridModel's published result")
# PGM node voltages for 1os2msr (tests/data/state_estimation/1os2msr/sym_output.json)
pgm = Dict(1 => (u=10751.072595758282, θ=-0.013054638926306409),
           2 => (u=10752.698591183394, θ=-0.017637349459726520),
           3 => (u=10748.320749959701, θ=-0.020182330759474040))
println(rpad("node",6), rpad("PGM |U| [V]",16), rpad("IL |U| [V]",16), rpad("NR |U| [V]",16), "max |U| err [V]")
voltage(res, nid) = (b = res.solution["bus"][string(nid)]; (b["vr"][1] + im*b["vi"][1]) * u_rated)
for nid in 1:3
    Uil = voltage(res_il, nid); Unr = voltage(res_nr, nid); Ug = pgm[nid].u * exp(im*pgm[nid].θ)
    println(rpad(nid,6), rpad(round(pgm[nid].u;digits=3),16), rpad(round(abs(Uil);digits=3),16),
            rpad(round(abs(Unr);digits=3),16), round(max(abs(Uil-Ug), abs(Unr-Ug)); sigdigits=2))
end

############################################################################ B
hr("PART B — the SAME solver on a FOUR-WIRE, explicit-neutral network")
#   PGM is single-phase (Kron-reduced).  Literal PMDSE keeps the neutral explicit:
#   every bus has terminals [1,2,3,4]; voltages are phase-to-neutral; a wye load's
#   neutral current is −Σ(phase currents).  Same `build_se_atoms`/`solve_se_*` code.

va = deg2rad.([0.0, -120.0, 120.0])
zp, zm, zn, zpn = 0.03+0.06im, 0.01+0.02im, 0.05+0.07im, 0.005+0.01im
Z = Matrix{ComplexF64}(undef, 4, 4)
for i in 1:3, j in 1:3; Z[i,j] = i == j ? zp : zm; end
Z[4,4] = zn; for i in 1:3; Z[i,4] = zpn; Z[4,i] = zpn; end
showmat("4×4 mutually-coupled series admittance  Y_branch = Z⁻¹  [p.u.]  (row/col 4 = neutral)", inv(Z))

busN(i,bt,gnd) = Dict{String,Any}("index"=>i,"bus_i"=>i,"bus_type"=>bt,"terminals"=>[1,2,3,4],
    "grounded"=>Bool[false,false,false,gnd],"vbase"=>1.0,"vmin"=>zeros(4),"vmax"=>fill(Inf,4),
    "vr_start"=>[cos.(va)...,0.0],"vi_start"=>[sin.(va)...,0.0])
mathEN = Dict{String,Any}(
    "bus"=>Dict("1"=>busN(1,3,true), "2"=>busN(2,1,false)),
    "branch"=>Dict("1"=>Dict{String,Any}("index"=>1,"f_bus"=>1,"t_bus"=>2,"f_connections"=>[1,2,3,4],
        "t_connections"=>[1,2,3,4],"br_r"=>real.(Z),"br_x"=>imag.(Z),"br_status"=>1)),
    "gen"=>Dict("1"=>Dict{String,Any}("index"=>1,"gen_bus"=>1,"connections"=>[1,2,3,4],"configuration"=>_PMD.WYE)),
    "load"=>Dict("1"=>Dict{String,Any}("index"=>1,"load_bus"=>2,"connections"=>[1,2,3,4],
        "configuration"=>_PMD.WYE,"pd"=>[0.3,0.4,0.5],"qd"=>[0.1,0.12,0.15],"model"=>_PMD.POWER,"status"=>1)),
    "shunt"=>Dict{String,Any}(), "settings"=>Dict{String,Any}("sbase"=>1.0))

# Ground truth by a quick Newton power flow (so we can check the estimator recovers it)
lmEN = PMDSE.LiteralModel(mathEN; reference = :prop); ni = lmEN.node_index; n = lmEN.n
Yen = complex.(lmEN.G, lmEN.B); Sload = [0.3+0.1im, 0.4+0.12im, 0.5+0.15im]
Uslk = ComplexF64[exp(im*va[1]), exp(im*va[2]), exp(im*va[3]), 0.0]
U = ComplexF64[(b == 1 ? Uslk[t] : (t < 4 ? exp(im*va[t]) : 0.0)) for (b,t) in lmEN.nodes]
fixed = [ni[(1,t)] for t in 1:4]; freeN = [i for i in 1:n if !(i in fixed)]
resid(x) = (Uu = copy(U); for (j,i) in enumerate(freeN); Uu[i] = x[j] + im*x[length(freeN)+j]; end;
            I = Yen*Uu; Ic = zeros(ComplexF64, n);
            for ph in 1:3; kc = ni[(2,ph)]; kn = ni[(2,4)]; Il = conj(Sload[ph]/(Uu[kc]-Uu[kn])); Ic[kc]-=Il; Ic[kn]+=Il; end;
            vcat(real.((Yen*Uu .- Ic)[freeN]), imag.((Yen*Uu .- Ic)[freeN])))
let x = vcat(real.(U[freeN]), imag.(U[freeN]))
    for _ in 1:60
        r = resid(x); J = zeros(length(x), length(x)); h = 1e-7
        for j in eachindex(x); xp = copy(x); xp[j] += h; J[:,j] = (resid(xp) .- r) ./ h; end
        d = J \ r; x .-= d; LinearAlgebra.norm(d, Inf) < 1e-13 && break
    end
    for (j,i) in enumerate(freeN); U[i] = x[j] + im*x[length(freeN)+j]; end
end
Utrue = copy(U)

# measurements: voltage phasor at the slack + per-phase current injections at the load
mEN = Dict{String,Any}(); kc = Ref(0); add!(d) = (kc[] += 1; mEN[string(kc[])] = d)
add!(Dict{String,Any}("cmp"=>:bus,"cmp_id"=>1,"var"=>:vr,"dst"=>[N(real(Utrue[ni[(1,p)]]),1e-4) for p in 1:3]))
add!(Dict{String,Any}("cmp"=>:bus,"cmp_id"=>1,"var"=>:vi,"dst"=>[N(imag(Utrue[ni[(1,p)]]),1e-4) for p in 1:3]))
crd = [conj(Sload[ph] / (Utrue[ni[(2,ph)]] - Utrue[ni[(2,4)]])) for ph in 1:3]   # load currents
add!(Dict{String,Any}("cmp"=>:load,"cmp_id"=>1,"var"=>:crd,"dst"=>[N(real(crd[p]),1e-4) for p in 1:3]))
add!(Dict{String,Any}("cmp"=>:load,"cmp_id"=>1,"var"=>:cid,"dst"=>[N(imag(crd[p]),1e-4) for p in 1:3]))
mathEN["meas"] = mEN

ref = (Dict(t=>real(Utrue[ni[(1,t)]]) for t in 1:4), Dict(t=>imag(Utrue[ni[(1,t)]]) for t in 1:4))
atomsEN = PMDSE.build_se_atoms(PMDSE.LiteralModel(mathEN; reference=:full_slack, ref_values=ref), mathEN)
println("\ncurrent-injection atoms at the load bus (terminal 4 is the neutral return = −Σ phases):")
for a in atomsEN
    a.kind == :cinj || continue
    nd = haskey(a.meta, :node) ? PMDSE.LiteralModel(mathEN; reference=:full_slack, ref_values=ref).nodes[a.meta.node] : ""
    println("   cinj at node ", nd, "   Ir=", round(a.z1; digits=3), "  Ii=", round(a.z2; digits=3))
end

for method in (:iterative_linear, :newton_raphson)
    r = PMDSE.solve_mc_se_literal(mathEN; estimator=:wls, method=method,
                                  reference=:full_slack, ref_values=ref, maxiter=100, tol=1e-12)
    err = maximum(abs((r.solution["bus"][string(b)]["vr"][findfirst(==(t), mathEN["bus"][string(b)]["terminals"])] +
                       im*r.solution["bus"][string(b)]["vi"][findfirst(==(t), mathEN["bus"][string(b)]["terminals"])]) -
                      Utrue[ni[(b,t)]]) for (b,t) in lmEN.nodes)
    println("\n", uppercase(string(method)), ": ", r.termination, " in ", r.iterations,
            " it,  max|U − truth| = ", round(err; sigdigits=3))
    b2 = r.solution["bus"]["2"]
    println("   bus-2 voltages |U| per terminal [1,2,3,N] = ", round.(b2["vm"]; digits=5),
            "   (neutral is non-zero: the 4-wire effect PGM cannot represent)")
end

############################################################################ C
hr("PART C — reference/gauge schemes and the robust estimators")
#   The PGM solve options (`method=:iterative_linear|:newton_raphson`) are the
#   general path and run on either network.  The reference scheme fixes the gauge:
#     :prop       — fix only grounded terminals; the rotation is taken from a
#                   voltage-phasor measurement, or pinned automatically (Im U_ref=0)
#                   when only magnitudes are measured.
#     :sota       — additionally ground the reference-bus neutral + one angle datum.
#     :full_slack — fix the whole reference-bus phasor.
println("reference / gauge schemes on the single-phase network (method=:iterative_linear):")
for r in (:prop, :sota)
    res = PMDSE.solve_mc_se_literal(math; estimator=:wls, method=:iterative_linear, reference=r)
    println("   ", rpad(string(r),11), " term=", res.termination, "  iters=", res.iterations)
end

#   The Gauss-Newton estimators :wls/:wlav/:mle (the original JuMP-free path,
#   selected by leaving `method` unset) target the explicit-neutral IVREN model,
#   so we exercise them on the four-wire network from Part B.  :wlav (least
#   absolute value) is robust to a single gross error; :mle reduces to :wls for
#   Gaussian noise.
println("\nGauss-Newton estimators on the four-wire EN network (method unset):  :wls / :wlav / :mle")
for est in (:wls, :wlav, :mle)
    res = PMDSE.solve_mc_se_literal(mathEN; estimator=est, reference=:full_slack, ref_values=ref)
    println("   ", rpad(string(est),5), " term=", res.termination, "  iters=", res.iterations,
            "  SSR=", round(res.objective; sigdigits=3))
end
println("\nDONE.  See docs/src/literal_pmdse_tutorial.md for the narrated version.")
