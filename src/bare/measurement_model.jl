################################################################################
#  Literal PMDSE                                                                #
#  measurement_model.jl : rectangular explicit-neutral measurement functions    #
#                         h(x) and assembly of the residual model.              #
#                                                                              #
#  Every supported measurement var-type (see src/core/measurement_conversion.jl #
#  for the IVR/IVREN authoritative definitions) is mapped to a closure of the    #
#  nodal state `x = [vr; vi]`.  Branch / load / gen currents are eliminated via  #
#  the bus admittance `Ybus` (nodal current `I = Ybus*U`) and the branch         #
#  admittance (branch current `I_fr = Yff*U_f + Yft*U_t`).                       #
#                                                                              #
#  EN conventions:                                                             #
#   * voltage / power measurements are phase-to-neutral (subtract `_N_IDX`)      #
#   * residuals are formed over `setdiff(connections, _N_IDX)` (neutral skipped) #
#   * a wye component's neutral current is `-sum(phase currents)`                #
################################################################################

"per-row evaluation context shared across the residual vector"
struct HCtx{T}
    vr::Vector{T}
    vi::Vector{T}
    Ir::Vector{T}   # real part of nodal injection Ybus*U
    Ii::Vector{T}   # imag part of nodal injection Ybus*U
end

"""
    SEModel

Assembled residual model.  `hfun(x_free) -> Vector` returns the model prediction
of every residual row; `z` are the observed values and `w` the diagonal weights
(`w_i = 1/(rescaler*σ_i)^2`).  `rowmeta` documents each row.
"""
struct SEModel
    lm::LiteralModel
    rowfns::Vector{Any}      # each maps HCtx -> scalar
    z::Vector{Float64}
    w::Vector{Float64}
    rowmeta::Vector{NamedTuple}
    brdata::Vector{Any}      # precomputed branch admittance blocks (for branch rows)
end

nrows(m::SEModel) = length(m.z)

# ---- helpers ---------------------------------------------------------------

"ordered non-neutral connections of a measurement's active connection list"
_nonneutral(active) = [c for c in active if c != _N_IDX]

"mean (value) and std (sigma) of a per-conductor distribution entry"
function _zsigma(dst_entry, σ_exact::Float64)
    if isa(dst_entry, _DST.Normal)
        return _DST.mean(dst_entry), _DST.std(dst_entry)
    elseif isa(dst_entry, Real)
        return Float64(dst_entry), σ_exact      # hard / fixed value
    else
        return _DST.mean(dst_entry), _DST.std(dst_entry)
    end
end

"active connections of a component for a given measurement var"
function _active_connections(math::Dict, cmp::Symbol, cmp_id::Int)
    if cmp == :bus
        return math["bus"][string(cmp_id)]["terminals"]
    elseif cmp == :load
        return math["load"][string(cmp_id)]["connections"]
    elseif cmp == :gen
        return math["gen"][string(cmp_id)]["connections"]
    elseif cmp == :branch
        return math["branch"][string(cmp_id)]["f_connections"]
    else
        error("Literal PMDSE: unsupported measurement component $(cmp)")
    end
end

"precompute branch admittance real/imag blocks and node indices"
function _branch_blocks(lm::LiteralModel)
    bd = Dict{Int,Any}()
    for (i, br) in lm.math["branch"]
        get(br, "br_status", 1) == 0 && continue
        fc = br["f_connections"]; tc = br["t_connections"]
        Z = Matrix{ComplexF64}(br["br_r"] .+ im .* br["br_x"]); Ybr = inv(Z)
        m = length(fc)
        Ysh = haskey(br, "g_fr") ? Matrix{ComplexF64}(br["g_fr"] .+ im .* br["b_fr"]) : zeros(ComplexF64, m, m)
        Yff = Ybr .+ Ysh; Yft = .-Ybr
        F = [lm.node_index[(br["f_bus"], c)] for c in fc]
        T = [lm.node_index[(br["t_bus"], c)] for c in tc]
        bd[br["index"]] = (Gff = real.(Yff), Bff = imag.(Yff), Gft = real.(Yft), Bft = imag.(Yft), F = F, T = T, fc = fc)
    end
    return bd
end

"from-side branch current (cr,ci) for connection position `a` (1-based into fc)"
function _branch_current(bd, ctx::HCtx, a::Int)
    Gff, Bff, Gft, Bft, F, T = bd.Gff, bd.Bff, bd.Gft, bd.Bft, bd.F, bd.T
    cr = zero(eltype(ctx.vr)); ci = zero(eltype(ctx.vr))
    @inbounds for b in eachindex(F)
        vrf = ctx.vr[F[b]]; vif = ctx.vi[F[b]]; vrt = ctx.vr[T[b]]; vit = ctx.vi[T[b]]
        # I = (Gff+iBff)(vrf+i vif) + (Gft+iBft)(vrt+i vit)
        cr += Gff[a, b] * vrf - Bff[a, b] * vif + Gft[a, b] * vrt - Bft[a, b] * vit
        ci += Bff[a, b] * vrf + Gff[a, b] * vif + Bft[a, b] * vrt + Gft[a, b] * vit
    end
    return cr, ci
end

# ---- model assembly --------------------------------------------------------

"""
    build_se_model(lm, math; rescaler=1.0, σ_exact=1e-8, zero_inj_sigma=1e-6)

Assemble the residual model from `math["meas"]`.  Current and power *injection*
measurements (`crd,cid,crg,cig,pd,qd,pg,qg,ptot,qtot,cmd,cmg,cad,cag`) are
aggregated to the nodal injection `Ybus*U`; this is exact at the solution and
handles buses hosting several components (e.g. multiple loads).  Zero-injection
pseudo-measurements (`z=0`) close the model at source-free terminals for
observability, matching the KCL equalities of `IVRENPowerModel`.
"""
function build_se_model(lm::LiteralModel, math::Dict; rescaler::Float64 = 1.0,
                        σ_exact::Float64 = 1e-8, zero_inj_sigma::Float64 = 1e-6,
                        σ_floor::Float64 = 1e-7)
    rowfns = Any[]; z = Float64[]; w = Float64[]; rowmeta = NamedTuple[]
    brdata = _branch_blocks(lm)

    push_row!(fn, zi, wi, meta) = (push!(rowfns, fn); push!(z, zi); push!(w, wi); push!(rowmeta, meta))
    # floor σ so that a measurement whose value is exactly 0 (e.g. vi at a
    # balanced reference bus) does not produce an infinite weight.
    wt(σ) = 1.0 / (rescaler * max(σ, σ_floor))^2

    # nodal current-injection accumulators (real & imag), keyed by node position
    inj_r = Dict{Int,Float64}(); inj_i = Dict{Int,Float64}()
    var_r = Dict{Int,Float64}(); var_i = Dict{Int,Float64}()
    touched = Set{Int}()
    # nodes that physically host a gen/load (so are NOT zero-injection)
    sourced = Set{Int}()
    for (_, ld) in math["load"], c in ld["connections"]
        push!(sourced, lm.node_index[(ld["load_bus"], c)])
    end
    for (_, g) in math["gen"], c in g["connections"]
        push!(sourced, lm.node_index[(g["gen_bus"], c)])
    end

    # accumulate a phase-only current measurement onto nodes (with neutral=-sum)
    function accumulate_current!(bus::Int, conns, dst, part::Symbol, sgn::Float64)
        nn = _nonneutral(conns)
        vals = Float64[]; sgs = Float64[]
        for (idx, c) in enumerate(nn)
            μ, σ = _zsigma(dst[idx], σ_exact)
            push!(vals, μ); push!(sgs, σ)
            k = lm.node_index[(bus, c)]
            d = part == :r ? inj_r : inj_i; v = part == :r ? var_r : var_i
            d[k] = get(d, k, 0.0) + sgn * μ
            v[k] = get(v, k, 0.0) + σ^2
            push!(touched, k)
        end
        if _N_IDX in conns                       # neutral return = -sum(phases)
            k = lm.node_index[(bus, _N_IDX)]
            d = part == :r ? inj_r : inj_i; v = part == :r ? var_r : var_i
            d[k] = get(d, k, 0.0) + sgn * (-sum(vals))
            v[k] = get(v, k, 0.0) + sum(abs2, sgs)
            push!(touched, k)
        end
    end

    for (m, meas) in math["meas"]
        var = meas["var"]; cmp = meas["cmp"]; cmp_id = meas["cmp_id"]
        dst = meas["dst"]
        active = _active_connections(math, cmp, cmp_id)
        nn = _nonneutral(active)

        if var in (:vr, :vi)                                   # native voltage
            for (idx, c) in enumerate(nn)
                k = lm.node_index[(cmp_id, c)]
                μ, σ = _zsigma(dst[idx], σ_exact)
                fn = var == :vr ? (ctx -> ctx.vr[k]) : (ctx -> ctx.vi[k])
                push_row!(fn, μ, wt(σ), (m = m, var = var, node = k))
            end
        elseif var in (:vm, :vmn)                              # |U_c - U_n|
            kn = lm.node_index[(cmp_id, _N_IDX)]
            for (idx, c) in enumerate(nn)
                k = lm.node_index[(cmp_id, c)]
                μ, σ = _zsigma(dst[idx], σ_exact)
                push_row!(ctx -> sqrt((ctx.vr[k]-ctx.vr[kn])^2 + (ctx.vi[k]-ctx.vi[kn])^2),
                          μ, wt(σ), (m = m, var = var, node = k))
            end
        elseif var == :va                                      # ∠(U_c - U_n)
            kn = lm.node_index[(cmp_id, _N_IDX)]
            for (idx, c) in enumerate(nn)
                k = lm.node_index[(cmp_id, c)]
                μ, σ = _zsigma(dst[idx], σ_exact)
                push_row!(ctx -> atan(ctx.vi[k]-ctx.vi[kn], ctx.vr[k]-ctx.vr[kn]),
                          μ, wt(σ), (m = m, var = var, node = k))
            end
        elseif var == :vll                                     # line-to-line |U_i - U_j|
            pairs = length(nn) > 2 ? [(1, 2), (2, 3), (3, 1)] : [(1, 2)]
            for (idx, (a, b)) in enumerate(pairs)
                ka = lm.node_index[(cmp_id, nn[a])]; kb = lm.node_index[(cmp_id, nn[b])]
                μ, σ = _zsigma(dst[idx], σ_exact)
                push_row!(ctx -> sqrt((ctx.vr[ka]-ctx.vr[kb])^2 + (ctx.vi[ka]-ctx.vi[kb])^2),
                          μ, wt(σ), (m = m, var = var))
            end
        elseif var in (:cr, :ci, :cm, :ca)                     # branch current (from-side)
            bd = brdata[cmp_id]
            for (idx, c) in enumerate(nn)
                a = findfirst(isequal(c), bd.fc)
                μ, σ = _zsigma(dst[idx], σ_exact)
                fn = if var == :cr;     ctx -> _branch_current(bd, ctx, a)[1]
                     elseif var == :ci; ctx -> _branch_current(bd, ctx, a)[2]
                     elseif var == :cm; ctx -> (cc = _branch_current(bd, ctx, a); sqrt(cc[1]^2 + cc[2]^2))
                     else               ctx -> (cc = _branch_current(bd, ctx, a); atan(cc[2], cc[1])) end
                push_row!(fn, μ, wt(σ), (m = m, var = var))
            end
        elseif var in (:p, :q)                                 # branch power flow (EN)
            bd = brdata[cmp_id]
            fbus = math["branch"][string(cmp_id)]["f_bus"]
            kn = lm.node_index[(fbus, _N_IDX)]
            for (idx, c) in enumerate(nn)
                a = findfirst(isequal(c), bd.fc); k = lm.node_index[(fbus, c)]
                μ, σ = _zsigma(dst[idx], σ_exact)
                fn = if var == :p
                    ctx -> (cc = _branch_current(bd, ctx, a); cc[1]*(ctx.vr[k]-ctx.vr[kn]) + cc[2]*(ctx.vi[k]-ctx.vi[kn]))
                else
                    ctx -> (cc = _branch_current(bd, ctx, a); -cc[2]*(ctx.vr[k]-ctx.vr[kn]) + cc[1]*(ctx.vi[k]-ctx.vi[kn]))
                end
                push_row!(fn, μ, wt(σ), (m = m, var = var))
            end
        elseif var in (:crd, :cid)                             # load current  (inj sign -)
            accumulate_current!(math["load"][string(cmp_id)]["load_bus"], active, dst,
                                var == :crd ? :r : :i, -1.0)
        elseif var in (:crg, :cig)                             # gen current   (inj sign +)
            accumulate_current!(math["gen"][string(cmp_id)]["gen_bus"], active, dst,
                                var == :crg ? :r : :i, +1.0)
        elseif var in (:pd, :qd, :pg, :qg, :ptot, :qtot)       # power injection (EN, aggregated)
            isgen = var in (:pg, :qg) || (var in (:ptot, :qtot) && cmp == :gen)
            bus = isgen ? math["gen"][string(cmp_id)]["gen_bus"] : math["load"][string(cmp_id)]["load_bus"]
            s = isgen ? 1.0 : -1.0
            kn = lm.node_index[(bus, _N_IDX)]
            ispow_p = var in (:pd, :pg, :ptot)
            if var in (:ptot, :qtot)
                μ, σ = _zsigma(dst[1], σ_exact)
                ks = [lm.node_index[(bus, c)] for c in nn]
                fn = ctx -> sum( ispow_p ?
                        s*(ctx.Ir[k]*(ctx.vr[k]-ctx.vr[kn]) + ctx.Ii[k]*(ctx.vi[k]-ctx.vi[kn])) :
                        s*(-ctx.Ii[k]*(ctx.vr[k]-ctx.vr[kn]) + ctx.Ir[k]*(ctx.vi[k]-ctx.vi[kn])) for k in ks)
                push_row!(fn, μ, wt(σ), (m = m, var = var))
            else
                for (idx, c) in enumerate(nn)
                    k = lm.node_index[(bus, c)]; μ, σ = _zsigma(dst[idx], σ_exact)
                    fn = ispow_p ?
                        (ctx -> s*(ctx.Ir[k]*(ctx.vr[k]-ctx.vr[kn]) + ctx.Ii[k]*(ctx.vi[k]-ctx.vi[kn]))) :
                        (ctx -> s*(-ctx.Ii[k]*(ctx.vr[k]-ctx.vr[kn]) + ctx.Ir[k]*(ctx.vi[k]-ctx.vi[kn])))
                    push_row!(fn, μ, wt(σ), (m = m, var = var, node = k))
                end
            end
        elseif var in (:cmd, :cmg, :cad, :cag)                 # |I| / ∠I injection (aggregated)
            isgen = var in (:cmg, :cag)
            bus = isgen ? math["gen"][string(cmp_id)]["gen_bus"] : math["load"][string(cmp_id)]["load_bus"]
            ismag = var in (:cmd, :cmg)
            for (idx, c) in enumerate(nn)
                k = lm.node_index[(bus, c)]; μ, σ = _zsigma(dst[idx], σ_exact)
                fn = ismag ? (ctx -> sqrt(ctx.Ir[k]^2 + ctx.Ii[k]^2)) : (ctx -> atan(ctx.Ii[k], ctx.Ir[k]))
                push_row!(fn, μ, wt(σ), (m = m, var = var, node = k))
            end
        else
            error("Literal PMDSE: measurement var :$(var) not supported")
        end
    end

    # ---- nodal current-injection rows -------------------------------------
    for k in lm.inj_nodes
        if k in touched
            zr = inj_r[k]; wr = wt(sqrt(max(var_r[k], σ_exact^2)))
            push_row!(let kk = k; ctx -> ctx.Ir[kk] end, zr, wr, (var = :inj_r, node = k))
            zi = inj_i[k]; wi = wt(sqrt(max(var_i[k], σ_exact^2)))
            push_row!(let kk = k; ctx -> ctx.Ii[kk] end, zi, wi, (var = :inj_i, node = k))
        elseif !(k in sourced)
            # genuine zero-injection terminal -> KCL pseudo-measurement (z = 0)
            push_row!(let kk = k; ctx -> ctx.Ir[kk] end, 0.0, wt(zero_inj_sigma), (var = :zinj_r, node = k))
            push_row!(let kk = k; ctx -> ctx.Ii[kk] end, 0.0, wt(zero_inj_sigma), (var = :zinj_i, node = k))
        end
        # sourced-but-untouched (e.g. power-only nodes): handled by power rows
    end

    return SEModel(lm, rowfns, z, w, rowmeta, Any[])
end

"evaluate the full residual-model prediction vector at free state `x_free`"
function predict(m::SEModel, x_free::AbstractVector{T}) where {T}
    lm = m.lm
    x = expand_state(lm, x_free)
    vr, vi = vrvi(lm, x)
    Ir, Ii = nodal_injection(lm, collect(vr), collect(vi))
    ctx = HCtx{T}(collect(vr), collect(vi), Ir, Ii)
    out = Vector{T}(undef, length(m.rowfns))
    @inbounds for i in eachindex(m.rowfns)
        out[i] = m.rowfns[i](ctx)
    end
    return out
end
