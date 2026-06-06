################################################################################
#  Literal PMDSE - explicit matrix-based distribution system state estimation   #
#                                                                              #
#  literal_core.jl : node index map, nodal admittance (Y_bus) assembly,         #
#                    reference/observability handling and flat start.           #
#                                                                              #
#  The "literal" estimator mirrors the 4-wire IVR explicit-neutral model        #
#  (`IVRENPowerModel`, see src/prob/se_en.jl) but keeps only the nodal bus       #
#  voltages `x = [vr; vi]` as state and solves the textbook normal equations     #
#  (Abur & Exposito) instead of building a JuMP model.  Branch / load / gen      #
#  currents are eliminated through the bus admittance matrix.                    #
################################################################################

"index of the neutral conductor (kept explicit in the 4-wire EN model)"
# (`_N_IDX` is already defined in the package main module)

"""
    LiteralModel

Container with everything the measurement model and the solvers need.  The full
real state is laid out as `x = [vr_1..vr_n , vi_1..vi_n]` where node `k`
(`1:n`) is the `k`-th `(bus, terminal)` pair in `nodes`.  The complex bus
admittance is stored split into real/imaginary parts `Ybus = G + im*B` so that
the nodal current injection `I = Ybus*U` is evaluated with `ForwardDiff`-friendly
real matrix products

    Ir = G*vr - B*vi ,   Ii = B*vr + G*vi .

# Fields
- `nodes`          : ordered vector of `(bus_id, terminal)` pairs
- `node_index`     : map `(bus_id, terminal) -> position in 1:n`
- `n`              : number of nodes
- `G`, `B`         : real / imaginary part of `Ybus`  (n×n, dense Float64)
- `ref_bus`        : id of the reference bus (`bus_type == 3`)
- `reference`      : reference scheme `Symbol` (`:full_slack`, `:sota`, `:prop`)
- `fixed_mask`     : `BitVector` of length `2n`, `true` where a state entry is fixed
- `x_fixed`        : full `2n` vector holding the fixed values (0 where free)
- `free_idx`       : indices (into `1:2n`) of the free state entries
- `inj_nodes`      : nodes (positions) that carry a nodal-current-injection equation
"""
struct LiteralModel
    math::Dict{String,Any}
    nodes::Vector{Tuple{Int,Int}}
    node_index::Dict{Tuple{Int,Int},Int}
    n::Int
    G::Matrix{Float64}
    B::Matrix{Float64}
    ref_bus::Int
    reference::Symbol
    fixed_mask::BitVector
    x_fixed::Vector{Float64}
    free_idx::Vector{Int}
    inj_nodes::Vector{Int}
end

"return the reference bus id (the unique `bus_type == 3` bus)"
function literal_ref_bus(math::Dict)
    refs = [bus["index"] for (_, bus) in math["bus"] if bus["bus_type"] == 3]
    isempty(refs) && error("Literal PMDSE: no reference bus (bus_type == 3) found")
    length(refs) > 1 && @warn "Literal PMDSE: multiple reference buses found, using $(refs[1])"
    return refs[1]
end

"build the ordered `(bus, terminal)` node list and the inverse lookup"
function literal_node_map(math::Dict)
    nodes = Tuple{Int,Int}[]
    for b in sort(collect(keys(math["bus"])); by = x -> parse(Int, x))
        bus = math["bus"][b]
        for t in bus["terminals"]
            push!(nodes, (bus["index"], t))
        end
    end
    node_index = Dict{Tuple{Int,Int},Int}(nd => k for (k, nd) in enumerate(nodes))
    return nodes, node_index
end

"""
    build_ybus(math, node_index, n) -> Matrix{ComplexF64}

Assemble the nodal admittance matrix of the explicit-neutral network.  Every
branch contributes its full (mutually-coupled) series admittance `Y = Z^{-1}`,
`Z = br_r + im*br_x`, stamped over the `f_connections`/`t_connections`
terminals, plus its Pi-model line shunts (`g_fr+im*b_fr`, `g_to+im*b_to`).  Bus
shunts (`math["shunt"]`) are added on the diagonal block.  This reproduces the
admittance implied by `constraint_mc_bus_voltage_drop` /
`constraint_mc_current_balance_se(::IVRENPowerModel, …)`.
"""
function build_ybus(math::Dict, node_index::Dict{Tuple{Int,Int},Int}, n::Int)
    Ybus = zeros(ComplexF64, n, n)

    for (_, br) in math["branch"]
        get(br, "br_status", 1) == 0 && continue
        f_bus = br["f_bus"]; t_bus = br["t_bus"]
        fc = br["f_connections"]; tc = br["t_connections"]
        Z = Matrix{ComplexF64}(br["br_r"] .+ im .* br["br_x"])
        Ybr = inv(Z)
        m = length(fc)
        Ysh_fr = haskey(br, "g_fr") ? Matrix{ComplexF64}(br["g_fr"] .+ im .* br["b_fr"]) : zeros(ComplexF64, m, m)
        Ysh_to = haskey(br, "g_to") ? Matrix{ComplexF64}(br["g_to"] .+ im .* br["b_to"]) : zeros(ComplexF64, m, m)

        F = [node_index[(f_bus, c)] for c in fc]
        T = [node_index[(t_bus, c)] for c in tc]

        Ybus[F, F] .+= Ybr .+ Ysh_fr
        Ybus[T, T] .+= Ybr .+ Ysh_to
        Ybus[F, T] .-= Ybr
        Ybus[T, F] .-= Ybr
    end

    for (_, sh) in get(math, "shunt", Dict{String,Any}())
        get(sh, "status", 1) == 0 && continue
        sb = sh["shunt_bus"]
        conns = sh["connections"]
        Ysh = Matrix{ComplexF64}(sh["gs"] .+ im .* sh["bs"])
        S = [node_index[(sb, c)] for c in conns]
        Ybus[S, S] .+= Ysh
    end

    return Ybus
end

"""
    reference_partition(math, nodes, node_index, n, ref_bus; reference, ref_values)

Decide which state entries are *fixed* (known parameters, removed as columns of
the Jacobian) versus *free*.  Returns `(fixed_mask, x_fixed)` of length `2n`.

Reference schemes (configurable, see §5 of the plan / `main.tex`):
- `:full_slack` : fix the whole reference-bus phasor (sanity check; forces a
  balanced reference bus).
- `:sota`       : ground the reference-bus neutral *and* fix one phase-angle
  datum `vi[ref, first_phase] = 0` (the SOTA combination that makes the gain
  full-rank without biasing the reference bus).
- `:prop`       : ground only the neutral(s); **rank-deficient by one** – used to
  reproduce the observability trap of `main.tex`.

Grounded terminals (`bus["grounded"]`) are always fixed to 0 in every scheme.
`ref_values` optionally provides `(vr, vi)` per reference-bus terminal (e.g. the
true slack phasor) for `:full_slack`; otherwise a balanced phasor scaled by the
bus base voltage start is used.
"""
function reference_partition(math::Dict, nodes, node_index, n::Int, ref_bus::Int;
                             reference::Symbol = :sota, ref_values = nothing)
    fixed_mask = falses(2n)
    x_fixed = zeros(Float64, 2n)

    # 1) grounded terminals -> hard zero everywhere
    for (_, bus) in math["bus"]
        for (idx, t) in enumerate(bus["terminals"])
            if bus["grounded"][idx]
                k = node_index[(bus["index"], t)]
                fixed_mask[k] = true;     x_fixed[k] = 0.0
                fixed_mask[n+k] = true;   x_fixed[n+k] = 0.0
            end
        end
    end

    rbus = math["bus"][string(ref_bus)]
    rterms = rbus["terminals"]
    phases = [t for t in rterms if t != _N_IDX]

    if reference == :full_slack
        # fix the full reference-bus phasor
        if ref_values === nothing
            va = deg2rad.([0.0, -120.0, 120.0])
            vr_ref = Dict{Int,Float64}(); vi_ref = Dict{Int,Float64}()
            for (j, t) in enumerate(phases)
                vr_ref[t] = cos(va[mod1(j, 3)]); vi_ref[t] = sin(va[mod1(j, 3)])
            end
            for t in rterms
                t == _N_IDX && (vr_ref[t] = 0.0; vi_ref[t] = 0.0)
            end
        else
            vr_ref, vi_ref = ref_values
        end
        for t in rterms
            k = node_index[(ref_bus, t)]
            fixed_mask[k] = true;   x_fixed[k]   = vr_ref[t]
            fixed_mask[n+k] = true; x_fixed[n+k] = vi_ref[t]
        end
    elseif reference == :sota
        # ground reference-bus neutral
        if _N_IDX in rterms
            k = node_index[(ref_bus, _N_IDX)]
            fixed_mask[k] = true;   x_fixed[k] = 0.0
            fixed_mask[n+k] = true; x_fixed[n+k] = 0.0
        end
        # fix one phase-angle datum: vi[ref, first phase] = 0
        kp = node_index[(ref_bus, first(phases))]
        fixed_mask[n+kp] = true; x_fixed[n+kp] = 0.0
    elseif reference == :prop
        # only ground reference-bus neutral (rank-deficient by one)
        if _N_IDX in rterms
            k = node_index[(ref_bus, _N_IDX)]
            fixed_mask[k] = true;   x_fixed[k] = 0.0
            fixed_mask[n+k] = true; x_fixed[n+k] = 0.0
        end
    else
        error("Literal PMDSE: unknown reference scheme :$(reference)")
    end

    return fixed_mask, x_fixed
end

"nodes (positions) that carry a nodal current-injection equation: every
ungrounded terminal that is **not** on the reference bus (the reference bus is a
free current slack)."
function injection_node_set(math::Dict, nodes, node_index, ref_bus::Int)
    grounded = Set{Tuple{Int,Int}}()
    for (_, bus) in math["bus"]
        for (idx, t) in enumerate(bus["terminals"])
            bus["grounded"][idx] && push!(grounded, (bus["index"], t))
        end
    end
    inj = Int[]
    for (k, (b, t)) in enumerate(nodes)
        (b == ref_bus) && continue
        ((b, t) in grounded) && continue
        push!(inj, k)
    end
    return inj
end

"""
    flat_start(math, nodes; vbase) -> Vector{Float64}

Balanced flat start: phases at `1∠{0,-120,120}`, neutral ≈ 0.  When the bus data
carries `vr_start`/`vi_start` (set by `_PMD.add_start_vrvi!`) those are used.
"""
function flat_start(math::Dict, nodes, node_index, n::Int)
    x = zeros(Float64, 2n)
    va = deg2rad.([0.0, -120.0, 120.0])
    for (k, (b, t)) in enumerate(nodes)
        bus = math["bus"][string(b)]
        if haskey(bus, "vr_start") && haskey(bus, "vi_start")
            idx = findfirst(isequal(t), bus["terminals"])
            x[k]   = bus["vr_start"][idx]
            x[n+k] = bus["vi_start"][idx]
        elseif t != _N_IDX
            j = findfirst(isequal(t), [1, 2, 3])
            ang = j === nothing ? 0.0 : va[j]
            x[k]   = cos(ang)
            x[n+k] = sin(ang)
        end
    end
    return x
end

"""
    LiteralModel(math; reference=:sota, ref_values=nothing)

Build the full literal model from a PMD *mathematical* data dictionary
(`data_math`, as produced by `_PMD.transform_data_model(...; kron_reduce=false,
phase_project=false)`).
"""
function LiteralModel(math::Dict; reference::Symbol = :sota, ref_values = nothing)
    nodes, node_index = literal_node_map(math)
    n = length(nodes)
    Ybus = build_ybus(math, node_index, n)
    ref_bus = literal_ref_bus(math)
    fixed_mask, x_fixed = reference_partition(math, nodes, node_index, n, ref_bus;
                                              reference = reference, ref_values = ref_values)
    free_idx = [i for i in 1:2n if !fixed_mask[i]]
    inj_nodes = injection_node_set(math, nodes, node_index, ref_bus)
    return LiteralModel(math, nodes, node_index, n, real.(Ybus), imag.(Ybus),
                        ref_bus, reference, fixed_mask, x_fixed, free_idx, inj_nodes)
end

"reconstruct the full `2n` real state from the free entries"
@inline function expand_state(lm::LiteralModel, x_free::AbstractVector{T}) where {T}
    x = convert(Vector{T}, lm.x_fixed)
    @inbounds for (j, i) in enumerate(lm.free_idx)
        x[i] = x_free[j]
    end
    return x
end

"split a full `2n` real state into `(vr, vi)` complex-part vectors of length n"
@inline function vrvi(lm::LiteralModel, x::AbstractVector)
    return @views x[1:lm.n], x[lm.n+1:2lm.n]
end

"nodal current injection `I = Ybus*U` split into real/imag parts (ForwardDiff ok)"
@inline function nodal_injection(lm::LiteralModel, vr::AbstractVector, vi::AbstractVector)
    Ir = lm.G * vr .- lm.B * vi
    Ii = lm.B * vr .+ lm.G * vi
    return Ir, Ii
end

"initial free-state vector from the (balanced) flat start"
function flat_start_free(lm::LiteralModel)
    x0 = flat_start(lm.math, lm.nodes, lm.node_index, lm.n)
    return x0[lm.free_idx]
end

"""
    state_solution(lm, x_full) -> Dict

Convert a full real state vector into a PMD-style solution dictionary
`Dict("bus" => Dict(id => Dict("vr"=>[...], "vi"=>[...])))`, with the per-bus
vectors ordered like `bus["terminals"]`.
"""
function state_solution(lm::LiteralModel, x_full::AbstractVector)
    vr, vi = vrvi(lm, x_full)
    busdict = Dict{String,Any}()
    for (b, bus) in lm.math["bus"]
        terms = bus["terminals"]
        vrb = [vr[lm.node_index[(bus["index"], t)]] for t in terms]
        vib = [vi[lm.node_index[(bus["index"], t)]] for t in terms]
        busdict[b] = Dict{String,Any}("vr" => vrb, "vi" => vib,
                                      "vm" => sqrt.(vrb .^ 2 .+ vib .^ 2))
    end
    return Dict{String,Any}("bus" => busdict)
end
