################################################################################
#  Shared helpers for the Literal PMDSE test suite.                              #
#  (Assumes the including scope has imported `_PMD`, `_PMDSE`, `_DST`,           #
#   `ForwardDiff`, `LinearAlgebra`, `Ipopt` and `Test`.)                         #
################################################################################

const LIT_N = _PMDSE._N_IDX   # neutral conductor index (4)

"main.tex toy: 3-bus 4-wire feeder with realistic per-unit impedance"
function toy_math()
    Zb = (400.0^2) / 100_000.0
    zp  = (0.01   + 0.02im)  / Zb
    zm  = (0.001  + 0.002im) / Zb
    zn  = (0.05   + 0.01im)  / Zb
    zpn = (0.0005 + 0.001im) / Zb
    Z = Matrix{ComplexF64}(undef, 4, 4)
    for i in 1:3, j in 1:3; Z[i, j] = i == j ? zp : zm; end
    Z[4, 4] = zn
    for i in 1:3; Z[i, 4] = zpn; Z[4, i] = zpn; end
    br_r = real.(Z); br_x = imag.(Z)
    va = deg2rad.([0.0, -120.0, 120.0])
    vr0 = [cos.(va)..., 0.0]; vi0 = [sin.(va)..., 0.0]
    bus(idx, bt, gnd) = Dict{String,Any}("index"=>idx, "bus_i"=>idx, "bus_type"=>bt,
        "terminals"=>[1,2,3,4], "grounded"=>Bool[false,false,false,gnd],
        "vbase"=>0.4/sqrt(3), "vmin"=>zeros(4), "vmax"=>fill(Inf,4),
        "vr_start"=>copy(vr0), "vi_start"=>copy(vi0))
    branch(idx, f, t) = Dict{String,Any}("index"=>idx, "f_bus"=>f, "t_bus"=>t,
        "f_connections"=>[1,2,3,4], "t_connections"=>[1,2,3,4],
        "br_r"=>br_r, "br_x"=>br_x, "br_status"=>1)
    load(idx, b, conns, pd, qd) = Dict{String,Any}("index"=>idx, "load_bus"=>b,
        "connections"=>conns, "configuration"=>_PMD.WYE, "pd"=>pd, "qd"=>qd,
        "model"=>_PMD.POWER, "status"=>1)
    Dict{String,Any}(
        "bus"=>Dict("1"=>bus(1,3,true), "2"=>bus(2,1,false), "3"=>bus(3,1,false)),
        "branch"=>Dict("1"=>branch(1,1,2), "2"=>branch(2,2,3)),
        "load"=>Dict("1"=>load(1,2,[1,2,3,4],[0.3,0.4,0.5],[0.1,0.1,0.2]),
                     "2"=>load(2,3,[1,4],[0.4],[0.1])),
        "gen"=>Dict("1"=>Dict{String,Any}("index"=>1,"gen_bus"=>1,
                     "connections"=>[1,2,3,4],"configuration"=>_PMD.WYE)),
        "shunt"=>Dict{String,Any}(), "settings"=>Dict{String,Any}("sbase"=>1.0))
end

"balanced reference-bus phasor for a math dict, keyed by terminal"
function balanced_ref(math)
    rb = _PMDSE.literal_ref_bus(math); va = deg2rad.([0.0,-120.0,120.0])
    vr = Dict{Int,Float64}(); vi = Dict{Int,Float64}()
    for t in math["bus"][string(rb)]["terminals"]
        vr[t] = t==LIT_N ? 0.0 : cos(va[t]); vi[t] = t==LIT_N ? 0.0 : sin(va[t])
    end
    (vr, vi)
end

"constant-power Newton PF for the toy; returns the free-state vector of `lm`"
function toy_pf_state(lm)
    math = lm.math; inj = lm.inj_nodes
    function comp_inj(vr, vi)
        Ic = Complex.(zeros(eltype(vr), lm.n))
        for (_, ld) in math["load"]
            b = ld["load_bus"]; conns = ld["connections"]; ph = _PMDSE._nonneutral(conns)
            kn = LIT_N in conns ? lm.node_index[(b,LIT_N)] : 0
            for (idx, c) in enumerate(ph)
                k = lm.node_index[(b,c)]
                Up = vr[k]+im*vi[k]; Un = kn==0 ? 0.0+0im : vr[kn]+im*vi[kn]
                Il = conj((ld["pd"][idx]+im*ld["qd"][idx])/(Up-Un))
                Ic[k] -= Il; kn != 0 && (Ic[kn] += Il)
            end
        end
        Ic
    end
    function resid(xf)
        x = _PMDSE.expand_state(lm, xf); vr,vi = _PMDSE.vrvi(lm,x)
        Ir,Ii = _PMDSE.nodal_injection(lm, collect(vr), collect(vi)); Ic = comp_inj(collect(vr), collect(vi))
        out = similar(xf, 2*length(inj))
        for (j,k) in enumerate(inj); out[2j-1]=Ir[k]-real(Ic[k]); out[2j]=Ii[k]-imag(Ic[k]); end
        out
    end
    xf = _PMDSE.flat_start_free(lm)
    for _ in 1:60
        F = resid(xf); J = ForwardDiff.jacobian(resid, xf); Δ = J \ F; xf .-= Δ
        LinearAlgebra.norm(Δ, Inf) < 1e-13 && break
    end
    xf
end

"toy ground-truth PF as a PMD-style solution dict (+ per-load pd/qd)"
function toy_pf()
    math = toy_math()
    lmf = _PMDSE.LiteralModel(math; reference=:full_slack, ref_values=balanced_ref(math))
    xf = toy_pf_state(lmf); sol = _PMDSE.state_solution(lmf, _PMDSE.expand_state(lmf, xf))
    for (b,bsol) in sol["bus"]; bsol["terminals"] = math["bus"][b]["terminals"]; end
    pf = Dict("solution"=>Dict("bus"=>sol["bus"], "load"=>Dict{String,Any}()))
    for (l,ld) in math["load"]; pf["solution"]["load"][l]=Dict("pd"=>ld["pd"],"qd"=>ld["qd"]); end
    math, pf
end

"parse + IVREN power-flow an explicit-neutral DSS feeder shipped with the package"
function load_en_feeder(name)
    eng = _PMD.parse_file(joinpath(_PMDSE.BASE_DIR,"test","data","three-bus-en-models",name))
    _PMD.transform_loops!(eng); _PMD.remove_all_bounds!(eng)
    math = _PMD.transform_data_model(eng, kron_reduce=false, phase_project=false)
    _PMD.add_start_vrvi!(math)
    slv = _PMDSE.optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0, "tol"=>1e-10)
    pf = _PMD.solve_mc_opf(math, _PMD.IVRENPowerModel, slv)
    for (b,bsol) in pf["solution"]["bus"]; bsol["terminals"]=math["bus"][b]["terminals"]; end
    math, pf
end

"write + add the canonical IVREN measurement set (vr/vi + crd/cid + crg/cig)"
function canonical_meas!(math, pf; σ=0.005)
    path = joinpath(mktempdir(),"m.csv")
    _PMDSE.write_measurements!(_PMD.IVRENPowerModel, math, pf, path, σ=σ)
    _PMDSE.add_measurements!(math, path, actual_meas=true)
    math
end

"reference-bus true phasor from a PF dict, as (vr_dict, vi_dict) over terminals"
function ref_values_from_pf(math, pf)
    rb = _PMDSE.literal_ref_bus(math)
    bus = math["bus"][string(rb)]; bsol = pf["solution"]["bus"][string(rb)]
    vr = Dict{Int,Float64}(); vi = Dict{Int,Float64}()
    for (idx,t) in enumerate(bus["terminals"]); vr[t]=bsol["vr"][idx]; vi[t]=bsol["vi"][idx]; end
    (vr, vi)
end

"complex node voltages from a PF dict, aligned with lm.nodes"
function pf_voltages(lm, pf)
    U = zeros(ComplexF64, lm.n)
    for (k,(b,t)) in enumerate(lm.nodes)
        bsol = pf["solution"]["bus"][string(b)]; idx=findfirst(isequal(t), lm.math["bus"][string(b)]["terminals"])
        U[k] = bsol["vr"][idx] + im*bsol["vi"][idx]
    end
    U
end

"true free-state for `lm` from a PF dict"
function pf_xfree(lm, pf)
    x = copy(lm.x_fixed)
    for (k,(b,t)) in enumerate(lm.nodes)
        bsol = pf["solution"]["bus"][string(b)]; idx=findfirst(isequal(t), lm.math["bus"][string(b)]["terminals"])
        x[k]=bsol["vr"][idx]; x[lm.n+k]=bsol["vi"][idx]
    end
    x[lm.free_idx]
end

"(gen − load) current injection per node, from PF component currents"
function pf_component_injection(lm, pf)
    enI(conns, cr, ci) = (np=length(cr); I=Complex{Float64}[cr[i]+im*ci[i] for i in 1:np];
                          length(conns)>np && push!(I, -sum(I)); I)
    Icmp = zeros(ComplexF64, lm.n)
    for (g,gen) in lm.math["gen"]
        Ig = enI(gen["connections"], pf["solution"]["gen"][g]["crg"], pf["solution"]["gen"][g]["cig"])
        for (idx,c) in enumerate(gen["connections"]); Icmp[lm.node_index[(gen["gen_bus"],c)]] += Ig[idx]; end
    end
    for (l,load) in lm.math["load"]
        Il = enI(load["connections"], pf["solution"]["load"][l]["crd"], pf["solution"]["load"][l]["cid"])
        for (idx,c) in enumerate(load["connections"]); Icmp[lm.node_index[(load["load_bus"],c)]] -= Il[idx]; end
    end
    Icmp
end

"a measurement dict exercising every supported var type (for the Jacobian test)"
function kitchen_sink_meas()
    n3() = [_DST.Normal(0.3*j, 0.01) for j in 1:3]; n1() = [_DST.Normal(0.2, 0.01)]
    M = Dict{String,Any}(); i = 0
    add(cmp, cid, var, dst) = (M["$(i+=1)"]=Dict{String,Any}("cmp"=>cmp,"cmp_id"=>cid,"var"=>var,"dst"=>dst,"crit"=>"wls"))
    for v in (:vr,:vi,:vm,:va); add(:bus,3,v,n3()); end
    add(:bus,3,:vll,n3())
    for v in (:cr,:ci,:cm,:ca,:p,:q); add(:branch,1,v,n3()); end
    for v in (:crd,:cid,:cmd,:cad,:pd,:qd); add(:load,1,v,n3()); end
    for v in (:crg,:cig,:cmg,:cag,:pg,:qg); add(:gen,1,v,n3()); end
    add(:load,1,:ptot,n1()); add(:load,1,:qtot,n1())
    M
end

"rotation-invariant measurement set (vm + pd + qd) for the observability test"
function rotation_invariant_meas(math, pf)
    M = Dict{String,Any}(); i = 0
    add(cmp, cid, var, dst) = (M["$(i+=1)"]=Dict{String,Any}("cmp"=>cmp,"cmp_id"=>cid,"var"=>var,"dst"=>dst,"crit"=>"wls"))
    for (b,bus) in math["bus"]
        bsol = pf["solution"]["bus"][b]; nt=length(bus["terminals"])
        vmn = [sqrt((bsol["vr"][k]-bsol["vr"][end])^2+(bsol["vi"][k]-bsol["vi"][end])^2) for k in 1:nt-1]
        add(:bus, bus["index"], :vm, [_DST.Normal(v,0.01) for v in vmn])
    end
    for (l,load) in math["load"]
        lsol = pf["solution"]["load"][l]; np=length(lsol["pd"])
        add(:load, load["index"], :pd, [_DST.Normal(lsol["pd"][k],0.01) for k in 1:np])
        add(:load, load["index"], :qd, [_DST.Normal(lsol["qd"][k],0.01) for k in 1:np])
    end
    M
end

"rotation direction δx=[-vi; vr] restricted to free coordinates"
function rotation_dir(lm, xfull)
    vr, vi = _PMDSE.vrvi(lm, xfull)
    d = zeros(2*lm.n); d[1:lm.n] = -vi; d[lm.n+1:2lm.n] = vr
    d[lm.free_idx]
end

with_meas(math, M) = (m = deepcopy(math); m["meas"] = M; m)
