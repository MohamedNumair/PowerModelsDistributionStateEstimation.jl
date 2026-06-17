################################################################################
#  Literal PMDSE                                                                #
#  pgm_se.jl : the two PowerGridModel (PGM) state-estimation *solve options*     #
#              re-implemented on the literal nodal model and made GENERAL        #
#              (four-wire, explicit-neutral capable).                            #
#                                                                              #
#  PGM (https://github.com/PowerGridModel/power-grid-model) solves the           #
#  weighted-least-squares DSSE with two interchangeable algorithms:             #
#                                                                              #
#    • iterative_linear  (PGM default) — every measurement is *linearised* to an #
#      equivalent complex-current / voltage-phasor measurement using the voltage #
#      angle from the previous iteration; a (constant) linear WLS system is then  #
#      re-solved until the voltages stop moving.                                #
#    • newton_raphson — the nonlinear WLS is solved directly by Gauss–Newton on  #
#      the real measurement model `z = h(x)+e`.                                  #
#                                                                              #
#  Both minimise   ½ (h(x)−z)ᵀ W (h(x)−z),   W = diag(1/σ²),                     #
#  i.e. they converge to the *same* state; PGM ships them as two methods and     #
#  this file mirrors that.  Unlike PGM (which is single-phase / Kron-reduced)    #
#  the implementation here keeps the full per-(bus,terminal) rectangular state   #
#  `x=[vr;vi]`, so it is general: an explicit neutral conductor is supported     #
#  (phase-to-neutral voltages/powers, neutral injection `= −Σ phase currents`),  #
#  and the single-phase PGM networks are simply the special case of one          #
#  conductor per bus with no neutral.                                           #
#                                                                              #
#  The measurement model is shared between the two methods through a list of     #
#  `SEAtom`s, each carrying the (real/imag parts of the) complex coefficient     #
#  vectors of the two linear functionals it needs:                              #
#      ΔU(x) = cu·U   (phase-to-neutral voltage)                                #
#      I(x)  = ci·U   (a branch or nodal-injection current, via Y)              #
#  From these every measurement type (voltage phasor/magnitude, branch power,    #
#  branch current, injection power/current, zero-injection) is expressed.        #
################################################################################

"weight given to a zero-injection KCL pseudo-measurement (1/σ²)"
const _ZINJ_W = 1.0e10

"""
    SEAtom

One scalar measurement (or one real/imaginary component of one complex
measurement) of the WLS problem, stored as the real and imaginary parts of the
complex coefficient vectors of its two linear functionals over the nodal complex
voltage `U` (length `n`):

* `cu = cur + im*cui` : the phase-to-neutral voltage `ΔU = cu·U`
  (used by voltage measurements and as the `U` in `S = ΔU·conj(I)`).
* `ci = cir + im*cii` : a current `I = ci·U` (a branch current row, or a
  `Y_bus` row for a nodal injection).

`kind`:
* `:vre` / `:vim` — measured real / imaginary part of `ΔU` (a voltage phasor
  component); one residual row, `z = z1`.
* `:vmag`        — measured magnitude `|ΔU|`; one (nonlinear) residual row for
  Newton–Raphson, two (linearised phasor) rows for iterative-linear.
* `:power`       — measured `(P,Q) = z1,z2` of `S = sign·ΔU·conj(I)`; two rows.
* `:cinj`        — measured complex current `I = z1+im*z2`; two rows.
* `:zinj`        — zero-injection constraint `I = 0`; two rows.
"""
Base.@kwdef struct SEAtom
    kind::Symbol
    cur::Vector{Float64} = Float64[]
    cui::Vector{Float64} = Float64[]
    cir::Vector{Float64} = Float64[]
    cii::Vector{Float64} = Float64[]
    z1::Float64 = 0.0
    z2::Float64 = 0.0
    sign::Float64 = 1.0
    sigma::Float64 = 1.0    # σ of a voltage / current measurement
    sigp::Float64 = 1.0     # σ of the active-power component
    sigq::Float64 = 1.0     # σ of the reactive-power component
    meta::NamedTuple = NamedTuple()
end

# ---- complex-coefficient helpers ------------------------------------------

"`(re,im)` of `Σ_k (cr[k]+im*ci[k])·(vr[k]+im*vi[k])` (ForwardDiff-friendly)"
@inline function _cval(cr::Vector{Float64}, ci::Vector{Float64}, vr, vi)
    re = zero(eltype(vr)); im_ = zero(eltype(vr))
    @inbounds for k in eachindex(cr)
        re  += cr[k] * vr[k] - ci[k] * vi[k]
        im_ += ci[k] * vr[k] + cr[k] * vi[k]
    end
    return re, im_
end

"same as `_cval` but for a fixed `ComplexF64` voltage vector `U`"
@inline function _cvalU(cr::Vector{Float64}, ci::Vector{Float64}, U::Vector{ComplexF64})
    s = zero(ComplexF64)
    @inbounds for k in eachindex(cr)
        s += (cr[k] + im * ci[k]) * U[k]
    end
    return s
end

"unit complex coefficient `e_k` (length n)"
_unit_coeff(k::Int, n::Int) = (v = zeros(Float64, n); v[k] = 1.0; v)

# ---- branch admittance blocks (from & to side) ----------------------------

"""
    _pgm_branch_blocks(lm) -> Dict

Per-branch full 2×2 (block) admittance: `Yff,Yft,Ytf,Ytt` (complex matrices over
the branch conductors), the from/to node positions `F,T`, the conductor list,
and the from/to neutral node positions (`Fn`/`Tn`, `nothing` if no neutral).
"""
function _pgm_branch_blocks(lm::LiteralModel)
    bd = Dict{Int,Any}()
    for (_, br) in lm.math["branch"]
        get(br, "br_status", 1) == 0 && continue
        fc = br["f_connections"]; tc = br["t_connections"]
        Z = Matrix{ComplexF64}(br["br_r"] .+ im .* br["br_x"]); Ybr = inv(Z)
        m = length(fc)
        Ysh_fr = haskey(br, "g_fr") ? Matrix{ComplexF64}(br["g_fr"] .+ im .* br["b_fr"]) : zeros(ComplexF64, m, m)
        Ysh_to = haskey(br, "g_to") ? Matrix{ComplexF64}(br["g_to"] .+ im .* br["b_to"]) : zeros(ComplexF64, m, m)
        F = [lm.node_index[(br["f_bus"], c)] for c in fc]
        T = [lm.node_index[(br["t_bus"], c)] for c in tc]
        Fn = haskey(lm.node_index, (br["f_bus"], _N_IDX)) ? lm.node_index[(br["f_bus"], _N_IDX)] : nothing
        Tn = haskey(lm.node_index, (br["t_bus"], _N_IDX)) ? lm.node_index[(br["t_bus"], _N_IDX)] : nothing
        bd[br["index"]] = (Yff = Ybr .+ Ysh_fr, Yft = .-Ybr, Ytf = .-Ybr, Ytt = Ybr .+ Ysh_to,
                           F = F, T = T, fc = fc, tc = tc,
                           f_bus = br["f_bus"], t_bus = br["t_bus"], Fn = Fn, Tn = Tn)
    end
    return bd
end

"neutral node position of `bus` (or `nothing` when the bus has no neutral)"
function _neutral_pos(lm::LiteralModel, bus::Int)
    haskey(lm.node_index, (bus, _N_IDX)) ? lm.node_index[(bus, _N_IDX)] : nothing
end

"phase-to-neutral coefficient `e_kc − e_kn` (or `e_kc` if no neutral) of length n"
function _delta_coeff(lm::LiteralModel, bus::Int, c::Int)
    n = lm.n
    cu = _unit_coeff(lm.node_index[(bus, c)], n)
    kn = _neutral_pos(lm, bus)
    kn !== nothing && c != _N_IDX && (cu[kn] -= 1.0)
    return cu
end

"""
    _kalman_combine(a, b) -> per-phase combined `Normal` distributions

Inverse-variance ("Kalman") combine of two per-conductor measurement vectors of
the *same* physical quantity (PGM §3.3): `1/σ² = Σ 1/σ_k²`,
`μ = σ² Σ μ_k/σ_k²`.  A disabled sensor (`σ → ∞`) contributes nothing.
"""
function _kalman_combine(a, b)
    return [begin
                μa, σa = _zsigma(a[i], 1.0e-8); μb, σb = _zsigma(b[i], 1.0e-8)
                ia = 1.0 / σa^2; ib = 1.0 / σb^2; iv = ia + ib
                iv > 0 ? _DST.Normal((μa * ia + μb * ib) / iv, sqrt(1.0 / iv)) : _DST.Normal(μa, Inf)
            end for i in eachindex(a)]
end

"store `dst` under `dict[key][part]`, Kalman-combining a repeated measurement"
function _accum_meas!(dict, key, part, dst)
    g = get!(dict, key, Dict{Symbol,Any}())
    g[part] = haskey(g, part) ? _kalman_combine(g[part], dst) : dst
    return nothing
end

"node position of the reference bus' first non-neutral terminal"
function _ref_phase_pos(lm::LiteralModel)
    terms = lm.math["bus"][string(lm.ref_bus)]["terminals"]
    ph = first(t for t in terms if t != _N_IDX)
    return lm.node_index[(lm.ref_bus, ph)]
end

# ---- measurement → atoms ---------------------------------------------------

"""
    build_se_atoms(lm, math; rescaler=1.0, σ_exact=1e-8, zinj_sigma=1e-5)

Translate `math["meas"]` (the PMD-style measurement dictionary, same one the
JuMP estimators consume) into the flat list of [`SEAtom`](@ref)s used by both
PGM solve options.  Appliance (`load`/`gen`) injection measurements are
aggregated onto the bus, exactly as PGM aggregates appliance powers into the bus
injection; current injections add the neutral return `= −Σ phase currents`.
Genuine zero-injection terminals get a `z=0` KCL pseudo-measurement.

Recognised `var`s (per non-neutral conductor unless noted):
`:vr`,`:vi` (phasor components) · `:vm`,`:vmn` (magnitude) ·
`:p`,`:q` (branch power, `meta["side"]` ∈ `(:from,:to)`, default `:from`) ·
`:cr`,`:ci` (branch current) · `:pd`,`:qd`,`:pg`,`:qg` (injection power) ·
`:crd`,`:cid`,`:crg`,`:cig` (injection current).
"""
function build_se_atoms(lm::LiteralModel, math::Dict; rescaler::Float64 = 1.0,
                        σ_exact::Float64 = 1.0e-8, zinj_sigma::Float64 = 1.0e-5)
    n = lm.n
    Ybus = complex.(lm.G, lm.B)
    bd = _pgm_branch_blocks(lm)
    atoms = SEAtom[]
    rs(σ) = rescaler * σ

    # accumulators for appliance injection aggregation, keyed by node position
    pw_S = Dict{Int,ComplexF64}();  pw_vp = Dict{Int,Float64}(); pw_vq = Dict{Int,Float64}()
    cu_I = Dict{Int,ComplexF64}();  cu_v = Dict{Int,Float64}()
    touched = Set{Int}()
    sourced = Set{Int}()
    for (_, ld) in get(math, "load", Dict{String,Any}()), c in ld["connections"]
        push!(sourced, lm.node_index[(ld["load_bus"], c)])
    end
    for (_, g) in get(math, "gen", Dict{String,Any}()), c in g["connections"]
        push!(sourced, lm.node_index[(g["gen_bus"], c)])
    end

    # paired power / current branch measurements, keyed by (cmp_id, side, part)
    bpow = Dict{Tuple{Int,Symbol},Dict{Symbol,Any}}()
    bcur = Dict{Tuple{Int,Symbol},Dict{Symbol,Any}}()
    # paired injection measurements, keyed by (cmp, cmp_id)
    ipow = Dict{Tuple{Symbol,Int},Dict{Symbol,Any}}()
    icur = Dict{Tuple{Symbol,Int},Dict{Symbol,Any}}()

    for (mid, meas) in math["meas"]
        var = meas["var"]; cmp = meas["cmp"]; cmp_id = meas["cmp_id"]; dst = meas["dst"]
        side = Symbol(get(meas, "side", :from))
        if var in (:vr, :vi)                                   # voltage phasor component
            bus = cmp_id
            for (idx, c) in enumerate(_nonneutral(_active_connections(math, cmp, cmp_id)))
                μ, σ = _zsigma(dst[idx], σ_exact)
                cu = _delta_coeff(lm, bus, c)
                push!(atoms, SEAtom(kind = (var == :vr ? :vre : :vim),
                                    cur = real(cu), cui = imag(cu),
                                    z1 = μ, sigma = rs(σ), meta = (mid = mid, var = var, c = c)))
            end
        elseif var in (:vm, :vmn)                              # voltage magnitude |ΔU|
            bus = cmp_id
            for (idx, c) in enumerate(_nonneutral(_active_connections(math, cmp, cmp_id)))
                μ, σ = _zsigma(dst[idx], σ_exact)
                cu = _delta_coeff(lm, bus, c)
                push!(atoms, SEAtom(kind = :vmag, cur = real(cu), cui = imag(cu),
                                    z1 = μ, sigma = rs(σ), meta = (mid = mid, var = var, c = c)))
            end
        elseif var == :p
            _accum_meas!(bpow, (cmp_id, side), :p, dst)
        elseif var == :q
            _accum_meas!(bpow, (cmp_id, side), :q, dst)
        elseif var == :cr
            _accum_meas!(bcur, (cmp_id, side), :r, dst)
        elseif var == :ci
            _accum_meas!(bcur, (cmp_id, side), :i, dst)
        elseif var in (:pd, :pg)
            _accum_meas!(ipow, (cmp, cmp_id), :p, dst)
        elseif var in (:qd, :qg)
            _accum_meas!(ipow, (cmp, cmp_id), :q, dst)
        elseif var in (:crd, :crg)
            _accum_meas!(icur, (cmp, cmp_id), :r, dst)
        elseif var in (:cid, :cig)
            _accum_meas!(icur, (cmp, cmp_id), :i, dst)
        else
            error("Literal PMDSE (PGM): measurement var :$(var) not supported by the PGM solve options")
        end
    end

    # ---- branch power measurements ----------------------------------------
    for ((cmp_id, side), pq) in bpow
        haskey(pq, :p) && haskey(pq, :q) || error("Literal PMDSE (PGM): branch $(cmp_id) $(side) power needs both :p and :q")
        b = bd[cmp_id]
        Yself = side == :from ? b.Yff : b.Ytt
        Yother = side == :from ? b.Yft : b.Ytf
        Fpos = side == :from ? b.F : b.T
        Opos = side == :from ? b.T : b.F
        sbus = side == :from ? b.f_bus : b.t_bus
        for (idx, c) in enumerate(_nonneutral(b.fc))
            a = findfirst(isequal(c), b.fc)
            ci = zeros(ComplexF64, n)
            @inbounds for q in eachindex(Fpos); ci[Fpos[q]] += Yself[a, q]; end
            @inbounds for q in eachindex(Opos); ci[Opos[q]] += Yother[a, q]; end
            cu = _delta_coeff(lm, sbus, c)
            μp, σp = _zsigma(pq[:p][idx], σ_exact); μq, σq = _zsigma(pq[:q][idx], σ_exact)
            push!(atoms, SEAtom(kind = :power, cur = real(cu), cui = imag(cu),
                                cir = real(ci), cii = imag(ci), z1 = μp, z2 = μq, sign = 1.0,
                                sigp = rs(σp), sigq = rs(σq), meta = (var = :p, branch = cmp_id, side = side, c = c)))
        end
    end

    # ---- branch current measurements --------------------------------------
    for ((cmp_id, side), ri) in bcur
        haskey(ri, :r) && haskey(ri, :i) || error("Literal PMDSE (PGM): branch $(cmp_id) $(side) current needs both :cr and :ci")
        b = bd[cmp_id]
        Yself = side == :from ? b.Yff : b.Ytt
        Yother = side == :from ? b.Yft : b.Ytf
        Fpos = side == :from ? b.F : b.T
        Opos = side == :from ? b.T : b.F
        for (idx, c) in enumerate(_nonneutral(b.fc))
            a = findfirst(isequal(c), b.fc)
            ci = zeros(ComplexF64, n)
            @inbounds for q in eachindex(Fpos); ci[Fpos[q]] += Yself[a, q]; end
            @inbounds for q in eachindex(Opos); ci[Opos[q]] += Yother[a, q]; end
            μr, σr = _zsigma(ri[:r][idx], σ_exact); μi, _ = _zsigma(ri[:i][idx], σ_exact)
            push!(atoms, SEAtom(kind = :cinj, cir = real(ci), cii = imag(ci),
                                z1 = μr, z2 = μi, sigma = rs(σr), meta = (var = :cr, branch = cmp_id, side = side, c = c)))
        end
    end

    # ---- injection power measurements (aggregate per node-conductor) -------
    for ((cmp, cmp_id), pq) in ipow
        haskey(pq, :p) && haskey(pq, :q) || error("Literal PMDSE (PGM): injection $(cmp) $(cmp_id) power needs both p and q")
        isgen = cmp == :gen
        bus = isgen ? math["gen"][string(cmp_id)]["gen_bus"] : math["load"][string(cmp_id)]["load_bus"]
        conns = isgen ? math["gen"][string(cmp_id)]["connections"] : math["load"][string(cmp_id)]["connections"]
        s = isgen ? 1.0 : -1.0
        for (idx, c) in enumerate(_nonneutral(conns))
            k = lm.node_index[(bus, c)]
            μp, σp = _zsigma(pq[:p][idx], σ_exact); μq, σq = _zsigma(pq[:q][idx], σ_exact)
            pw_S[k]  = get(pw_S, k, 0.0 + 0im) + s * (μp + im * μq)
            pw_vp[k] = get(pw_vp, k, 0.0) + (rs(σp))^2
            pw_vq[k] = get(pw_vq, k, 0.0) + (rs(σq))^2
            push!(touched, k)
        end
    end
    for (k, S) in pw_S
        cu = _unit_coeff(k, n)
        # phase-to-neutral: subtract the neutral of the bus owning node k
        (bus, c) = lm.nodes[k]; kn = _neutral_pos(lm, bus)
        kn !== nothing && c != _N_IDX && (cu[kn] -= 1.0)
        ci = collect(@view Ybus[k, :])
        push!(atoms, SEAtom(kind = :power, cur = real(cu), cui = imag(cu),
                            cir = real(ci), cii = imag(ci), z1 = real(S), z2 = imag(S), sign = 1.0,
                            sigp = sqrt(pw_vp[k]), sigq = sqrt(pw_vq[k]), meta = (var = :pinj, node = k)))
    end

    # ---- injection current measurements (aggregate + neutral return) ------
    for ((cmp, cmp_id), ri) in icur
        haskey(ri, :r) && haskey(ri, :i) || error("Literal PMDSE (PGM): injection $(cmp) $(cmp_id) current needs both cr and ci")
        isgen = cmp == :gen
        bus = isgen ? math["gen"][string(cmp_id)]["gen_bus"] : math["load"][string(cmp_id)]["load_bus"]
        conns = isgen ? math["gen"][string(cmp_id)]["connections"] : math["load"][string(cmp_id)]["connections"]
        s = isgen ? 1.0 : -1.0
        Isum = 0.0 + 0im; vsum = 0.0
        for (idx, c) in enumerate(_nonneutral(conns))
            k = lm.node_index[(bus, c)]
            μr, σr = _zsigma(ri[:r][idx], σ_exact); μi, _ = _zsigma(ri[:i][idx], σ_exact)
            Iph = s * (μr + im * μi)
            cu_I[k] = get(cu_I, k, 0.0 + 0im) + Iph
            cu_v[k] = get(cu_v, k, 0.0) + (rs(σr))^2
            push!(touched, k)
            Isum += Iph; vsum += (rs(σr))^2
        end
        if _N_IDX in conns                                     # neutral return = −Σ phase currents
            kn = lm.node_index[(bus, _N_IDX)]
            cu_I[kn] = get(cu_I, kn, 0.0 + 0im) - Isum
            cu_v[kn] = get(cu_v, kn, 0.0) + vsum
            push!(touched, kn)
        end
    end
    for (k, I) in cu_I
        ci = collect(@view Ybus[k, :])
        push!(atoms, SEAtom(kind = :cinj, cir = real(ci), cii = imag(ci),
                            z1 = real(I), z2 = imag(I), sigma = sqrt(cu_v[k]), meta = (var = :cinj, node = k)))
    end

    # ---- zero-injection KCL pseudo-measurements ---------------------------
    for k in lm.inj_nodes
        (k in touched || k in sourced) && continue
        ci = collect(@view Ybus[k, :])
        push!(atoms, SEAtom(kind = :zinj, cir = real(ci), cii = imag(ci),
                            sigma = zinj_sigma, meta = (var = :zinj, node = k)))
    end

    return atoms
end

# ---- shared reference / gauge handling ------------------------------------

"the free indices and fixed state for a solve, adding the slack-angle gauge
`Im(U_ref)=0` when the measurements carry no absolute angle information"
function _gauge(lm::LiteralModel, atoms::Vector{SEAtom})
    has_angle = any(a -> a.kind == :vim, atoms)
    fixed = copy(lm.fixed_mask)
    if !has_angle
        fixed[lm.n + _ref_phase_pos(lm)] = true     # pin the global rotation
    end
    free = [i for i in 1:2lm.n if !fixed[i]]
    return free, copy(lm.x_fixed), has_angle
end

"reconstruct the full `2n` state (ForwardDiff-friendly) from the free entries"
function _expand_free(xfix::Vector{Float64}, free::Vector{Int}, xf::AbstractVector{T}) where {T}
    x = convert(Vector{T}, xfix)
    @inbounds for (j, i) in enumerate(free); x[i] = xf[j]; end
    return x
end

"complex flat-start voltages for `lm` (balanced phasors, neutral ≈ 0)"
function _flat_U(lm::LiteralModel)
    x = flat_start(lm.math, lm.nodes, lm.node_index, lm.n)
    return complex.(x[1:lm.n], x[lm.n+1:2lm.n])
end

# ---- iterative-linear method ----------------------------------------------

"""
    solve_se_il(lm, atoms; maxiter=100, tol=1e-9, verbose=false)

PowerGridModel `iterative_linear` state estimation on the literal model.  Each
iteration re-linearises the power/current and voltage-magnitude measurements at
the latest voltages (only the right-hand side changes; the measurement matrix is
constant) and solves the linear WLS `min ‖√W (A x − b)‖₂` for the new
rectangular state `x=[vr;vi]`, until `max_i |Uᵢ−Uᵢ_prev| < tol`.
"""
function solve_se_il(lm::LiteralModel, atoms::Vector{SEAtom}; maxiter::Int = 100,
                     tol::Float64 = 1.0e-9, verbose::Bool = false)
    t0 = time()
    n = lm.n
    free, xfix, _ = _gauge(lm, atoms)
    U = _flat_U(lm)
    # seed fixed nodes (e.g. a `:full_slack` reference bus) to their known phasor
    @inbounds for i in 1:n
        lm.fixed_mask[i] && (U[i] = xfix[i] + im * xfix[n+i])
    end
    term = :maxiter; iters = 0

    # the measurement matrix A (constant) — rhs/weights are rebuilt per iteration
    rows = Vector{Float64}[]
    for a in atoms
        if a.kind == :vre
            push!(rows, vcat(a.cur, -a.cui))
        elseif a.kind == :vim
            push!(rows, vcat(a.cui, a.cur))
        elseif a.kind == :vmag
            push!(rows, vcat(a.cur, -a.cui)); push!(rows, vcat(a.cui, a.cur))
        elseif a.kind in (:power, :cinj, :zinj)
            push!(rows, vcat(a.cir, -a.cii)); push!(rows, vcat(a.cii, a.cir))
        end
    end
    A = permutedims(reduce(hcat, rows))                       # m × 2n
    Afree = A[:, free]
    Afix_b = A * xfix                                        # fixed-column contribution
    w = _il_weights(atoms)

    for it in 1:maxiter
        iters = it
        b = _il_rhs(atoms, U)
        beff = b .- Afix_b
        sw = sqrt.(w)
        xf = (sw .* Afree) \ (sw .* beff)
        xfull = copy(xfix); @inbounds for (j, i) in enumerate(free); xfull[i] = xf[j]; end
        Unew = complex.(xfull[1:n], xfull[n+1:2n])
        dev = maximum(abs.(Unew .- U))
        U = Unew
        verbose && println("  il it $it  max|ΔU|=$(dev)")
        if dev < tol; term = :converged; break; end
    end

    x_full = vcat(real.(U), imag.(U))
    obj, gcond, grank = _atoms_diagnostics(atoms, lm, x_full, free, xfix)
    sol = state_solution(lm, x_full)
    return LiteralResult(term, iters, obj, time() - t0, gcond, grank,
                         x_full[free], x_full, sol, :iterative_linear)
end

"per-row weights of the iterative-linear measurement matrix (constant)"
function _il_weights(atoms::Vector{SEAtom})
    w = Float64[]
    for a in atoms
        if a.kind in (:vre, :vim)
            push!(w, 1.0 / a.sigma^2)
        elseif a.kind == :vmag
            push!(w, 1.0 / a.sigma^2); push!(w, 1.0 / a.sigma^2)
        elseif a.kind == :power
            wS = 1.0 / (a.sigp^2 + a.sigq^2)                 # PGM: σ_S² = σ_P² + σ_Q²
            push!(w, wS); push!(w, wS)
        elseif a.kind == :cinj
            push!(w, 1.0 / a.sigma^2); push!(w, 1.0 / a.sigma^2)
        elseif a.kind == :zinj
            push!(w, _ZINJ_W); push!(w, _ZINJ_W)
        end
    end
    return w
end

"right-hand side of the iterative-linear system, re-linearised at voltages `U`"
function _il_rhs(atoms::Vector{SEAtom}, U::Vector{ComplexF64})
    b = Float64[]
    for a in atoms
        if a.kind in (:vre, :vim)
            push!(b, a.z1)
        elseif a.kind == :vmag
            du = _cvalU(a.cur, a.cui, U)
            tgt = a.z1 * (abs(du) > 0 ? du / abs(du) : 1.0 + 0im)   # use previous angle
            push!(b, real(tgt)); push!(b, imag(tgt))
        elseif a.kind == :power
            du = _cvalU(a.cur, a.cui, U)
            Imeas = conj(a.sign * (a.z1 + im * a.z2) / du)          # S → equivalent current
            push!(b, real(Imeas)); push!(b, imag(Imeas))
        elseif a.kind == :cinj
            push!(b, a.z1); push!(b, a.z2)
        elseif a.kind == :zinj
            push!(b, 0.0); push!(b, 0.0)
        end
    end
    return b
end

# ---- newton-raphson method ------------------------------------------------

"the nonlinear measurement prediction `h(x)` over the free state (ForwardDiff ok)"
function _nr_predict(atoms::Vector{SEAtom}, n::Int, xfix::Vector{Float64},
                     free::Vector{Int}, xf::AbstractVector{T}) where {T}
    x = _expand_free(xfix, free, xf)
    vr = @view x[1:n]; vi = @view x[n+1:2n]
    h = T[]
    for a in atoms
        if a.kind == :vre
            re, _ = _cval(a.cur, a.cui, vr, vi); push!(h, re)
        elseif a.kind == :vim
            _, im_ = _cval(a.cur, a.cui, vr, vi); push!(h, im_)
        elseif a.kind == :vmag
            re, im_ = _cval(a.cur, a.cui, vr, vi); push!(h, sqrt(re^2 + im_^2))
        elseif a.kind == :power
            dur, dui = _cval(a.cur, a.cui, vr, vi); ir, ii = _cval(a.cir, a.cii, vr, vi)
            push!(h, a.sign * (dur * ir + dui * ii)); push!(h, a.sign * (dui * ir - dur * ii))
        elseif a.kind in (:cinj, :zinj)
            ir, ii = _cval(a.cir, a.cii, vr, vi); push!(h, ir); push!(h, ii)
        end
    end
    return h
end

"measured values `z` and weights `w` aligned with `_nr_predict` rows"
function _nr_zw(atoms::Vector{SEAtom})
    z = Float64[]; w = Float64[]
    for a in atoms
        if a.kind in (:vre, :vim)
            push!(z, a.z1); push!(w, 1.0 / a.sigma^2)
        elseif a.kind == :vmag
            push!(z, a.z1); push!(w, 1.0 / a.sigma^2)
        elseif a.kind == :power
            push!(z, a.z1); push!(w, 1.0 / a.sigp^2)
            push!(z, a.z2); push!(w, 1.0 / a.sigq^2)
        elseif a.kind == :cinj
            push!(z, a.z1); push!(w, 1.0 / a.sigma^2)
            push!(z, a.z2); push!(w, 1.0 / a.sigma^2)
        elseif a.kind == :zinj
            push!(z, 0.0); push!(w, _ZINJ_W); push!(z, 0.0); push!(w, _ZINJ_W)
        end
    end
    return z, w
end

"""
    solve_se_nr(lm, atoms; maxiter=50, tol=1e-9, warm=true, verbose=false)

PowerGridModel `newton_raphson` state estimation: Gauss–Newton on the nonlinear
WLS `min ½‖√W (z − h(x))‖₂²`, `H = ∂h/∂x` via `ForwardDiff`, with the same
QR/orthogonal step as `solve_wls`.  By default the iteration is *warm-started*
from one `iterative_linear` solve (`warm=true`); the flat start is degenerate for
power-/magnitude-only systems with no phasor measurement (the angle Jacobian
vanishes), so warm-starting makes Newton–Raphson as robust as PGM's augmented
formulation while converging to the identical WLS optimum.
"""
function solve_se_nr(lm::LiteralModel, atoms::Vector{SEAtom}; maxiter::Int = 50,
                     tol::Float64 = 1.0e-9, warm::Bool = true, verbose::Bool = false)
    t0 = time()
    n = lm.n
    free, xfix, _ = _gauge(lm, atoms)
    z, w = _nr_zw(atoms)
    sqrtW = sqrt.(w)

    if warm
        x_full0 = solve_se_il(lm, atoms; maxiter = maxiter, tol = tol).x_full
        xf = x_full0[free]
    else
        xf = _flat_U(lm) |> U -> vcat(real.(U), imag.(U))[free]
    end

    term = :maxiter; iters = 0
    for it in 1:maxiter
        iters = it
        h = _nr_predict(atoms, n, xfix, free, xf)
        r = z .- h
        H = ForwardDiff.jacobian(xx -> _nr_predict(atoms, n, xfix, free, xx), xf)
        G = Symmetric_full(transpose(H) * (w .* H))
        rhs = transpose(H) * (w .* r)
        Δ, _ = _solve_gain(G, rhs, sqrtW .* H, sqrtW .* r)
        xf = xf .+ Δ
        verbose && println("  nr it $it  ‖Δ‖∞=$(LinearAlgebra.norm(Δ, Inf))")
        if LinearAlgebra.norm(Δ, Inf) < tol; term = :converged; break; end
    end

    x_full = _expand_free(xfix, free, xf)
    h = _nr_predict(atoms, n, xfix, free, xf); r = z .- h
    obj = sum(w .* r .^ 2)
    H = ForwardDiff.jacobian(xx -> _nr_predict(atoms, n, xfix, free, xx), xf)
    gcond, grank = _gain_diag(transpose(H) * (w .* H))
    sol = state_solution(lm, x_full)
    return LiteralResult(term, iters, obj, time() - t0, gcond, grank, xf, x_full, sol, :newton_raphson)
end

"weighted SSR + gain diagnostics of an atom set at a full state"
function _atoms_diagnostics(atoms::Vector{SEAtom}, lm::LiteralModel, x_full::Vector{Float64},
                            free::Vector{Int}, xfix::Vector{Float64})
    z, w = _nr_zw(atoms)
    xf = x_full[free]
    h = _nr_predict(atoms, lm.n, xfix, free, xf)
    obj = sum(w .* (z .- h) .^ 2)
    H = ForwardDiff.jacobian(xx -> _nr_predict(atoms, lm.n, xfix, free, xx), xf)
    gcond, grank = _gain_diag(transpose(H) * (w .* H))
    return obj, gcond, grank
end
