export solve_acp_mc_se_oltc, build_mc_se_oltc

"solves the AC state estimation with OLTC tap estimation"
function solve_acp_mc_se_oltc(data::Union{Dict{String,<:Any},String}, solver; kwargs...)
    return solve_mc_se_oltc(data, _PMD.ACPUPowerModel, solver; kwargs...)
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
function build_mc_se_oltc(pm::_PMD.AbstractUnbalancedPowerModel)

    # Variables
    _PMDSE.variable_mc_bus_voltage(pm, bounded = true)
    _PMD.variable_mc_branch_power(pm; bounded = true)
    _PMD.variable_mc_transformer_power(pm; bounded = true)
    _PMD.variable_mc_oltc_transformer_tap(pm)    # --- ADDED: Tap Estimation Variable ---
    _PMD.variable_mc_generator_power(pm; bounded = true)
    variable_mc_load(pm; report = true)
    variable_mc_residual(pm; bounded = true)
    variable_mc_measurement(pm; bounded = false)

    # Constraints
    for (i,gen) in _PMD.ref(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, i)
    end
    for (i,bus) in _PMD.ref(pm, :ref_buses)
        @assert bus["bus_type"] == 3
        _PMD.constraint_mc_theta_ref(pm, i)
    end
    for (i,bus) in _PMD.ref(pm, :bus)
        _PMDSE.constraint_mc_power_balance_se(pm, i)
    end
    for (i,branch) in _PMD.ref(pm, :branch)
        _PMD.constraint_mc_ohms_yt_from(pm, i)
        _PMD.constraint_mc_ohms_yt_to(pm,i)
    end
    for (i,meas) in _PMD.ref(pm, :meas)
        constraint_mc_residual(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
        _PMD.constraint_mc_transformer_power(pm, i)
    end

    # Objective
    objective_mc_se(pm)
end