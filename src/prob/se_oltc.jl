export solve_ivren_mc_se_oltc,solve_mc_se_oltc, build_mc_se_oltc

"solves the AC state estimation with OLTC tap estimation"
function solve_ivren_mc_se_oltc(data::Union{Dict{String,<:Any},String}, solver; kwargs...)
    return solve_mc_se_oltc(data, _PMD.IVRENPowerModel, solver; kwargs...)
end

"Internal solver function for OLTC SE"
function solve_mc_se_oltc(data::Union{Dict{String,<:Any},String}, model_type::Type, solver; kwargs...)
    if haskey(data["se_settings"], "criterion")
        _PMDSE.assign_unique_individual_criterion!(data)
    end
    if !haskey(data["se_settings"], "rescaler")
        data["se_settings"]["rescaler"] = 1
    end
    if !haskey(data["se_settings"], "number_of_gaussian")
        data["se_settings"]["number_of_gaussian"] = 10
    end
    return _PMD.solve_mc_model(data, model_type, solver, build_mc_se_oltc; kwargs...)
end

"Specification of the SE problem including Transformer Taps as variables"
function build_mc_se_oltc(pm::_PMD.IVRENPowerModel)

    # Variables
    _PMDSE.variable_mc_bus_voltage(pm, bounded = true)
    _PMD.variable_mc_branch_current(pm; bounded = true)
    _PMD.variable_mc_generator_current(pm, bounded = true)
    _PMD.variable_mc_transformer_current(pm; bounded = true)
    variable_mc_transformer_tap(pm)    # --- ADDED: Tap Estimation Variable ---
    variable_mc_load_current(pm; report = true)
    variable_mc_residual(pm; bounded = true)
    variable_mc_measurement(pm; bounded = false)


    for i in _PMD.ids(pm, :bus)
        if i in _PMD.ids(pm, :ref_buses)
        _PMD.constraint_mc_voltage_reference(pm, i)  # vm is not fixed
        end
    end
    
    for id in _PMD.ids(pm, :gen)
        constraint_mc_generator_current_se(pm, id)
    end

    for i in _PMD.ids(pm, :transformer)
        constraint_mc_transformer_tap_equal_phase(pm, i)
        constraint_mc_transformer_voltage(pm, i,fix_taps=false) 
        constraint_mc_transformer_current(pm, i,fix_taps=false)
    end


    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_current_from(pm, i)
        _PMD.constraint_mc_current_to(pm, i)
        _PMD.constraint_mc_bus_voltage_drop(pm, i)
    end

    for (i,bus) in _PMD.ref(pm, :bus)
        constraint_mc_current_balance_se(pm, i)
        #constraint_mc_neutral_grounding(pm, i)  #TODO: make it only grounded if load is grounded
    end

    for (i,meas) in _PMD.ref(pm, :meas)
        constraint_mc_residual(pm, i)
    end


    objective_mc_se(pm)
end

function variable_mc_transformer_tap(pm::_PMD.IVRENPowerModel;
    nw::Int=_PMD.nw_id_default, bounded::Bool=true, report::Bool=true
)
    @debug "Using OLTC Tap Variables in State Estimation Problem Formulation"
    # p_oltc_ids = [id for (id, tr) in _PMD.ref(pm, nw, :transformer)
    #              if endswith(string(get(tr, "source_id", "")), ".2")]
 
    # Only transformers for which NOT everything is fixed (i.e., Bool[0,0,0] -> variables; Bool[1,1,1] -> skip)
    # p_oltc_var_ids = [i for i in p_oltc_ids
    #                   if !all(_PMD.ref(pm, nw, :transformer, i, "tm_fix"))]

     p_oltc_ids = [id for (id,trans) in _PMD.ref(pm, nw, :transformer) if !all(trans["tm_fix"])]

    tap = _PMD.var(pm, nw)[:tap] = Dict(i => JuMP.@variable(pm.model,
        [p in 1:length(_PMD.ref(pm, nw, :transformer, i, "tm_set"))],
        base_name="$(nw)_tm_$(i)",
        start = _PMD.ref(pm, nw, :transformer, i, "tm_set")[p],
    ) for i in p_oltc_ids)
 
    if bounded
        for tr_id in p_oltc_ids, p in 1:length(_PMD.ref(pm, nw, :transformer, tr_id, "tm_set"))
            _PMD.set_lower_bound(_PMD.var(pm, nw)[:tap][tr_id][p], _PMD.ref(pm, nw, :transformer, tr_id, "tm_lb")[p])
            _PMD.set_upper_bound(_PMD.var(pm, nw)[:tap][tr_id][p], _PMD.ref(pm, nw, :transformer, tr_id, "tm_ub")[p])
        end
    end
 
    report && _IM.sol_component_value(pm, :pmd, nw, :transformer, :tap, p_oltc_ids, tap)
end



function constraint_mc_transformer_voltage(pm::_PMD.IVRENPowerModel, i::Int; nw::Int=_PMD.nw_id_default, fix_taps::Bool=true)
    transformer = _PMD.ref(pm, nw, :transformer, i)
    f_bus = transformer["f_bus"]
    t_bus = transformer["t_bus"]
    f_idx = (i, f_bus, t_bus)
    t_idx = (i, t_bus, f_bus)
    configuration = transformer["configuration"]
    f_connections = transformer["f_connections"]
    t_connections = transformer["t_connections"]
    tm_set = transformer["tm_set"]
    tm_fixed = fix_taps ? ones(Bool, length(tm_set)) : transformer["tm_fix"]
    tm_scale = _PMD.calculate_tm_scale(transformer, _PMD.ref(pm, nw, :bus, f_bus), _PMD.ref(pm, nw, :bus, t_bus))

    #TODO change data model
    # there is redundancy in specifying polarity seperately on from and to side
    #TODO change this once migrated to new data model
    pol = transformer["polarity"]

    if configuration == _PMD.WYE
        constraint_mc_transformer_voltage_yy(pm, nw, i, f_bus, t_bus, f_idx, t_idx, f_connections, t_connections, pol, tm_set, tm_fixed, tm_scale)
    elseif configuration == _PMD.DELTA
        constraint_mc_transformer_voltage_dy(pm, nw, i, f_bus, t_bus, f_idx, t_idx, f_connections, t_connections, pol, tm_set, tm_fixed, tm_scale)
    elseif configuration == "zig-zag"
        error("Zig-zag not yet supported.")
    end
end

function constraint_mc_transformer_voltage_yy(pm::_PMD.IVRENPowerModel, nw::Int, trans_id::Int, f_bus::Int, t_bus::Int, f_idx::Tuple{Int,Int,Int}, t_idx::Tuple{Int,Int,Int}, f_connections::Vector{Int}, t_connections::Vector{Int}, pol::Int, tm_set::Vector{<:Real}, tm_fixed::Vector{Bool}, tm_scale::Real)
    vr_fr_P = [_PMD.var(pm, nw, :vr, f_bus)[c] for c in f_connections[1:end-1]]
    vi_fr_P = [_PMD.var(pm, nw, :vi, f_bus)[c] for c in f_connections[1:end-1]]
    vr_fr_n = _PMD.var(pm, nw, :vr, f_bus)[f_connections[end]]
    vi_fr_n = _PMD.var(pm, nw, :vi, f_bus)[f_connections[end]]
    vr_to_P = [_PMD.var(pm, nw, :vr, t_bus)[c] for c in t_connections[1:end-1]]
    vi_to_P = [_PMD.var(pm, nw, :vi, t_bus)[c] for c in t_connections[1:end-1]]
    vr_to_n = _PMD.var(pm, nw, :vr, t_bus)[t_connections[end]]
    vi_to_n = _PMD.var(pm, nw, :vi, t_bus)[t_connections[end]]
    
    # construct tm as a parameter or scaled variable depending on whether it is fixed or not
    tm = [tm_fixed[idx] ? tm_set[idx] : _PMD.var(pm, nw, :tap, trans_id)[idx] for idx in 1:length(tm_fixed)]
    scale = (tm_scale*pol).*tm

    JuMP.@constraint(pm.model, (vr_fr_P.-vr_fr_n) .== scale.*(vr_to_P.-vr_to_n))
    JuMP.@constraint(pm.model, (vi_fr_P.-vi_fr_n) .== scale.*(vi_to_P.-vi_to_n))
end
function constraint_mc_transformer_voltage_dy(pm::_PMD.IVRENPowerModel, nw::Int, trans_id::Int, f_bus::Int, t_bus::Int, f_idx::Tuple{Int,Int,Int}, t_idx::Tuple{Int,Int,Int}, f_connections::Vector{Int}, t_connections::Vector{Int}, pol::Int, tm_set::Vector{<:Real}, tm_fixed::Vector{Bool}, tm_scale::Real)
    vr_fr_P = [_PMD.var(pm, nw, :vr, f_bus)[c] for c in f_connections]
    vi_fr_P = [_PMD.var(pm, nw, :vi, f_bus)[c] for c in f_connections]
    vr_to_P = [_PMD.var(pm, nw, :vr, t_bus)[c] for c in t_connections[1:end-1]]
    vi_to_P = [_PMD.var(pm, nw, :vi, t_bus)[c] for c in t_connections[1:end-1]]
    vr_to_n = _PMD.var(pm, nw, :vr, t_bus)[t_connections[end]]
    vi_to_n = _PMD.var(pm, nw, :vi, t_bus)[t_connections[end]]

    # construct tm as a parameter or scaled variable depending on whether it is fixed or not
    tm = [tm_fixed[idx] ? tm_set[idx] : _PMD.var(pm, nw, :tap, trans_id)[idx] for idx in 1:length(tm_fixed)]
    scale = (tm_scale*pol).*tm

    n_phases = length(tm)
    Md = _PMD._get_delta_transformation_matrix(n_phases)

    JuMP.@constraint(pm.model, Md*vr_fr_P .== scale.*(vr_to_P .- vr_to_n))
    JuMP.@constraint(pm.model, Md*vi_fr_P .== scale.*(vi_to_P .- vi_to_n))
end



function constraint_mc_transformer_current(pm::_PMD.IVRENPowerModel, i::Int; nw::Int=_IM.nw_id_default, fix_taps::Bool=true)
    # if ref(pm, nw_id_default, :conductors)!=3
    #     error("Transformers only work with networks with three conductors.")
    # end

    transformer = _PMD.ref(pm, nw, :transformer, i)
    f_bus = transformer["f_bus"]
    t_bus = transformer["t_bus"]
    f_idx = (i, f_bus, t_bus)
    t_idx = (i, t_bus, f_bus)
    configuration = transformer["configuration"]
    f_connections = transformer["f_connections"]
    t_connections = transformer["t_connections"]
    tm_set = transformer["tm_set"]
    tm_fixed = fix_taps ? ones(Bool, length(tm_set)) : transformer["tm_fix"]
    tm_scale = _PMD.calculate_tm_scale(transformer, _PMD.ref(pm, nw, :bus, f_bus), _PMD.ref(pm, nw, :bus, t_bus))

    #TODO change data model
    # there is redundancy in specifying polarity seperately on from and to side
    #TODO change this once migrated to new data model
    pol = transformer["polarity"]
    if configuration == _PMD.WYE
        constraint_mc_transformer_current_yy(pm, nw, i, f_bus, t_bus, f_idx, t_idx, f_connections, t_connections, pol, tm_set, tm_fixed, tm_scale)
    elseif configuration == _PMD.DELTA
        constraint_mc_transformer_current_dy(pm, nw, i, f_bus, t_bus, f_idx, t_idx, f_connections, t_connections, pol, tm_set, tm_fixed, tm_scale)
    elseif configuration == "zig-zag"
        error("Zig-zag not yet supported.")
    end
end

function constraint_mc_transformer_current_yy(pm::_PMD.IVRENPowerModel, nw::Int, trans_id::Int, f_bus::Int, t_bus::Int, f_idx::Tuple{Int,Int,Int}, t_idx::Tuple{Int,Int,Int}, f_connections::Vector{Int}, t_connections::Vector{Int}, pol::Int, tm_set::Vector{<:Real}, tm_fixed::Vector{Bool}, tm_scale::Real)
    cr_fr_P = _PMD.var(pm, nw, :crt, f_idx)
    ci_fr_P = _PMD.var(pm, nw, :cit, f_idx)
    cr_to_P = _PMD.var(pm, nw, :crt, t_idx)
    ci_to_P = _PMD.var(pm, nw, :cit, t_idx)
    
    # construct tm as a parameter or scaled variable depending on whether it is fixed or not
    tm = [tm_fixed[idx] ? tm_set[idx] : _PMD.var(pm, nw, :tap, trans_id)[idx] for idx in 1:length(tm_fixed)]
    scale = (tm_scale*pol).*tm

    JuMP.@constraint(pm.model, scale.*cr_fr_P .+ cr_to_P .== 0)
    JuMP.@constraint(pm.model, scale.*ci_fr_P .+ ci_to_P .== 0)

    _PMD.var(pm, nw, :crt_bus)[f_idx] = _PMD._merge_bus_flows(pm, [cr_fr_P..., -sum(cr_fr_P)], f_connections)
    _PMD.var(pm, nw, :cit_bus)[f_idx] = _PMD._merge_bus_flows(pm, [ci_fr_P..., -sum(ci_fr_P)], f_connections)
    _PMD.var(pm, nw, :crt_bus)[t_idx] = _PMD._merge_bus_flows(pm, [cr_to_P..., -sum(cr_to_P)], t_connections)
    _PMD.var(pm, nw, :cit_bus)[t_idx] = _PMD._merge_bus_flows(pm, [ci_to_P..., -sum(ci_to_P)], t_connections)
end

function constraint_mc_transformer_current_dy(pm::_PMD.IVRENPowerModel, nw::Int, trans_id::Int, f_bus::Int, t_bus::Int, f_idx::Tuple{Int,Int,Int}, t_idx::Tuple{Int,Int,Int}, f_connections::Vector{Int}, t_connections::Vector{Int}, pol::Int, tm_set::Vector{<:Real}, tm_fixed::Vector{Bool}, tm_scale::Real)
    cr_fr_P = _PMD.var(pm, nw, :crt, f_idx)
    ci_fr_P = _PMD.var(pm, nw, :cit, f_idx)
    cr_to_P = _PMD.var(pm, nw, :crt, t_idx)
    ci_to_P = _PMD.var(pm, nw, :cit, t_idx)
    
    # construct tm as a parameter or scaled variable depending on whether it is fixed or not
    tm = [tm_fixed[idx] ? tm_set[idx] : _PMD.var(pm, nw, :tap, trans_id)[idx] for idx in 1:length(tm_fixed)]
    scale = (tm_scale*pol).*tm

    n_phases = length(tm)
    Md = _PMD._get_delta_transformation_matrix(n_phases)

    JuMP.@constraint(pm.model, scale.*cr_fr_P .+ cr_to_P .== 0)
    JuMP.@constraint(pm.model, scale.*ci_fr_P .+ ci_to_P .== 0)

    _PMD.var(pm, nw, :crt_bus)[f_idx] = _PMD._merge_bus_flows(pm, Md'*cr_fr_P, f_connections)
    _PMD.var(pm, nw, :cit_bus)[f_idx] = _PMD._merge_bus_flows(pm, Md'*ci_fr_P, f_connections)
    _PMD.var(pm, nw, :crt_bus)[t_idx] = _PMD._merge_bus_flows(pm, [cr_to_P..., -sum(cr_to_P)], t_connections)
    _PMD.var(pm, nw, :cit_bus)[t_idx] = _PMD._merge_bus_flows(pm, [ci_to_P..., -sum(ci_to_P)], t_connections)
end
    
"Enforces equal tap across phases for transformer i (only if a tap variable exists)"
function constraint_mc_transformer_tap_equal_phase(
    pm::_PMD.AbstractUnbalancedPowerModel,
    i::Int;
    nw::Int=_PMD.nw_id_default
)
    tap_dict = get(_PMD.var(pm, nw), :tap, nothing)
    (tap_dict === nothing || !haskey(tap_dict, i)) && return

    tap = tap_dict[i]  # vector (per phase)
    for p in eachindex(tap)
        if p > 1
            JuMP.@constraint(pm.model, tap[p] == tap[1])
        end
    end
end