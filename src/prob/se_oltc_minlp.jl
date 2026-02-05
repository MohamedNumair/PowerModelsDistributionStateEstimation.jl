export solve_ivren_mc_se_oltc_minlp, solve_mc_se_oltc_minlp, build_mc_se_oltc_minlp

"solves the AC state estimation with OLTC tap estimation (MINLP formulation)"
function solve_ivren_mc_se_oltc_minlp(data::Union{Dict{String,<:Any},String}, solver; kwargs...)
    return solve_mc_se_oltc_minlp(data, _PMD.IVRENPowerModel, solver; kwargs...)
end

"Internal solver function for OLTC SE (MINLP)"
function solve_mc_se_oltc_minlp(data::Union{Dict{String,<:Any},String}, model_type::Type, solver; kwargs...)
    if haskey(data["se_settings"], "criterion")
        _PMDSE.assign_unique_individual_criterion!(data)
    end
    if !haskey(data["se_settings"], "rescaler")
        data["se_settings"]["rescaler"] = 1
    end
    if !haskey(data["se_settings"], "number_of_gaussian")
        data["se_settings"]["number_of_gaussian"] = 10
    end
    return _PMD.solve_mc_model(data, model_type, solver, build_mc_se_oltc_minlp; kwargs...)
end

"Specification of the SE problem including Transformer Taps as integer variables"
function build_mc_se_oltc_minlp(pm::_PMD.IVRENPowerModel)

    # Variables
    _PMDSE.variable_mc_bus_voltage(pm, bounded = true)
    _PMD.variable_mc_branch_current(pm; bounded = true)
    _PMD.variable_mc_generator_current(pm, bounded = true)
    _PMD.variable_mc_transformer_current(pm; bounded = true)
    variable_mc_transformer_tap_integer(pm)    # --- ADDED: Integer Tap Estimation Variable ---
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
        constraint_mc_transformer_voltage(pm, i, fix_taps=false) 
        constraint_mc_transformer_current(pm, i, fix_taps=false)
    end


    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_current_from(pm, i)
        _PMD.constraint_mc_current_to(pm, i)
        _PMD.constraint_mc_bus_voltage_drop(pm, i)
    end

    for (i,bus) in _PMD.ref(pm, :bus)
        constraint_mc_current_balance_se(pm, i)
    end

    for (i,meas) in _PMD.ref(pm, :meas)
        constraint_mc_residual(pm, i)
    end

    objective_mc_se(pm)
end

function variable_mc_transformer_tap_integer(pm::_PMD.IVRENPowerModel;
    nw::Int=_PMD.nw_id_default, bounded::Bool=true, report::Bool=true
)
    @debug "Using Integer OLTC Tap Variables in State Estimation Problem Formulation"

     p_oltc_ids = [id for (id,trans) in _PMD.ref(pm, nw, :transformer) if !all(trans["tm_fix"])]

    # In MINLP, we define an integer variable representing the step position
    # The actual tap value is then calculated as: tap_val = tm_nominal + step_size * step_int
    tap_int = _PMD.var(pm, nw)[:tap_int] = Dict(i => JuMP.@variable(pm.model,
        [p in 1:length(_PMD.ref(pm, nw, :transformer, i, "tm_set"))],
        base_name="$(nw)_tm_int_$(i)",
        integer = true
    ) for i in p_oltc_ids)
    
    # We also need a continuous expression/variable to be used in the voltage equations
    tap = _PMD.var(pm, nw)[:tap] = Dict(i => JuMP.@variable(pm.model,
        [p in 1:length(_PMD.ref(pm, nw, :transformer, i, "tm_set"))],
        base_name="$(nw)_tm_val_$(i)"
    ) for i in p_oltc_ids)

    for i in p_oltc_ids
        tr = _PMD.ref(pm, nw, :transformer, i)
        tm_set = tr["tm_set"]
        tm_step = get(tr, "tm_step", 0.0125) # Default step size usually 1.25%
        tm_min = tr["tm_lb"]
        tm_max = tr["tm_ub"]

        # If we assume tm_set contains the current tap position, we can calculate bounds for the integer steps
        # relative to 1.0 or the nominal tap.
        # Often simpler: tap = 1.0 + step * integer_var
        # integer_var bounds roughly (tm_min - 1.0)/step to (tm_max - 1.0)/step
        
        # Assuming nominal tap is approximately 1.0 for bounds calculation
        for p in 1:length(tm_set)
             # Calculate integer bounds
             lb_int = round(Int, (tm_min[p] - 1.0) / tm_step)
             ub_int = round(Int, (tm_max[p] - 1.0) / tm_step)
             
             if bounded
                 JuMP.set_lower_bound(tap_int[i][p], lb_int)
                 JuMP.set_upper_bound(tap_int[i][p], ub_int)
             end

             # Link integer step to continuous tap value
             # tap_val = 1.0 + step * tap_int
             JuMP.@constraint(pm.model, tap[i][p] == 1.0 + tm_step * tap_int[i][p])
             
             # Apply bounds to the mapped continuous variable as well, just in case
             if bounded
                JuMP.set_lower_bound(tap[i][p], tm_min[p])
                JuMP.set_upper_bound(tap[i][p], tm_max[p])
             end
        end
    end
 
    report && _IM.sol_component_value(pm, :pmd, nw, :transformer, :tap, p_oltc_ids, tap)
    report && _IM.sol_component_value(pm, :pmd, nw, :transformer, :tap_int, p_oltc_ids, tap_int)
end