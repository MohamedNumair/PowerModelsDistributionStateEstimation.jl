################################################################################
#  Copyright 2020, Marta Vanin, Tom Van Acker                                  #
################################################################################
# PowerModelsDistributionStateEstimation.jl                                    #
# An extention package of PowerModels(Distribution).jl for Static Power System #
# State Estimation.                                                            #
################################################################################
"""
    constraint_mc_residual

Equality constraint that describes the residual definition, which depends on the
criterion assigned to each individual measurement in data["meas"]["m"]["crit"].
"""
function constraint_mc_residual(pm::_PMD.AbstractUnbalancedPowerModel, i::Int; nw::Int=_IM.nw_id_default)

    cmp_id = get_cmp_id(pm, nw, i)
    res = _PMD.var(pm, nw, :res, i)
    var = _PMD.var(pm, nw, _PMD.ref(pm, nw, :meas, i, "var"), cmp_id)
    dst = _PMD.ref(pm, nw, :meas, i, "dst")
    rsc = _PMD.ref(pm, nw, :se_settings)["rescaler"]
    crit = _PMD.ref(pm, nw, :meas, i, "crit")
    meas_var = _PMD.ref(pm, nw, :meas, i, "var")
    conns = get_active_connections(pm, nw, _PMD.ref(pm, nw, :meas, i, "cmp"), cmp_id, meas_var)
    for (idx, c) in enumerate(setdiff(conns,[_N_IDX]))
        if (occursin("ls", crit) || occursin("lav", crit)) && isa(dst[idx], _DST.Normal)
            μ, σ = occursin("w", crit) ? (_DST.mean(dst[idx]), _DST.std(dst[idx])) : (_DST.mean(dst[idx]), 1.0)
        end
        if isa(dst[idx], Float64)
            JuMP.@constraint(pm.model, var[c] == dst[idx])
            JuMP.@constraint(pm.model, res[idx] == 0.0)         
        elseif crit ∈ ["wls", "ls"] && isa(dst[idx], _DST.Normal)
            JuMP.@constraint(pm.model,
                res[idx] * rsc^2 * σ^2 == (var[c] - μ)^2 
            )
        elseif crit == "rwls" && isa(dst[idx], _DST.Normal)
            JuMP.@constraint(pm.model,
                res[idx] * rsc^2 * σ^2 >= (var[c] - μ)^2
            )
        elseif crit ∈ ["wlav", "lav"] && isa(dst[idx], _DST.Normal)
            JuMP.@constraint(pm.model,
                res[idx] * rsc * σ == abs(var[c] - μ)
            )
        elseif crit == "rwlav" && isa(dst[idx], _DST.Normal)
            JuMP.@constraint(pm.model,
                res[idx] * rsc * σ >= (var[c] - μ) 
            )
            JuMP.@constraint(pm.model,
                res[idx] * rsc * σ >= - (var[c] - μ)
            )
        elseif crit == "mle"
            #TODO: enforce min and max in the meas dictionary and just with haskey make it optional for extendedbeta
            pkg_id = any([ dst[idx] isa d for d in [ExtendedBeta{Float64}, _Poly.Polynomial]]) ? _PMDSE : _DST
            lb = ( !isa(dst[idx], _DST.MixtureModel) && !isinf(pkg_id.minimum(dst[idx])) ) ? pkg_id.minimum(dst[idx]) : -10
            ub = ( !isa(dst[idx], _DST.MixtureModel) && !isinf(pkg_id.maximum(dst[idx])) ) ? pkg_id.maximum(dst[idx]) : 10
            if any([ dst[idx] isa d for d in [_DST.MixtureModel, _Poly.Polynomial]]) lb = _PMD.ref(pm, nw, :meas, i, "min") end
            if any([ dst[idx] isa d for d in [_DST.MixtureModel, _Poly.Polynomial]]) ub = _PMD.ref(pm, nw, :meas, i, "max") end
            
            shf = abs(Optim.optimize(x -> -pkg_id.logpdf(dst[idx],x),lb,ub).minimum)
            f = Symbol("df_",i,"_",c)

            fun(x) = rsc * ( - shf + pkg_id.logpdf(dst[idx],x) )
            grd(x) = pkg_id.gradlogpdf(dst[idx],x)
            hes(x) = heslogpdf(dst[idx],x)
            JuMP.register(pm.model, f, 1, fun, grd, hes)
            JuMP.add_nonlinear_constraint(pm.model, :($(res[idx]) == - $(f)($(var[c]))))
        else
            error("SE criterion of measurement $(i) not recognized")
        end
    end
end

# Constraints related to the ANGULAR REFERENCE MODELS
########################################################
## ACP
function variable_mc_bus_voltage(pm::_PMD.AbstractUnbalancedPowerModel; bounded = true)
    _PMD.variable_mc_bus_voltage_magnitude_only(pm; bounded = bounded)
    _PMDSE.variable_mc_bus_voltage_angle(pm; bounded = bounded)
end


"""
    variable_mc_bus_voltage_angle(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_PMD.nw_id_default, bounded::Bool=true, report::Bool=true)

Defines the bus voltage angle variables for a multi-conductor unbalanced power model.

# Arguments
- `pm::AbstractUnbalancedPowerModel`: The power model instance.
- `nw::Int`: The network identifier (default is `_PMD.nw_id_default`).
- `bounded::Bool`: If `true`, applies bounds to the voltage angle variables based on the bus data (default is `true`).
- `report::Bool`: If `true`, reports the solution component values (default is `true`).

# Description
This function initializes the bus voltage angle variables for each bus in the power model. It sets the starting values for the voltage angles based on default values or specified start values in the bus data. If `bounded` is `true`, it applies lower and upper bounds to the voltage angle variables based on the `vamin` and `vamax` fields in the bus data. If `report` is `true`, it reports the solution component values for the voltage angles.

# Notes
- The starting values for the voltage angles are converted from degrees to radians.
"""
function variable_mc_bus_voltage_angle(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_PMD.nw_id_default, bounded::Bool=true, report::Bool=true)
    terminals = Dict(i => bus["terminals"] for (i,bus) in _PMD.ref(pm, nw, :bus))
    va_start_defaults = Dict(i => deg2rad.([0.0, -120.0, 120.0, fill(0.0, length(terms))...][terms]) for (i, terms) in terminals)
    va = _PMD.var(pm, nw)[:va] = Dict(i => JuMP.@variable(pm.model,
            [t in terminals[i]], base_name="$(nw)_va_$(i)",
            start = _PMD.comp_start_value(_PMD.ref(pm, nw, :bus, i), ["va_start", "va"], t, va_start_defaults[i][findfirst(isequal(t), terminals[i])]),
            
        ) for i in _PMD.ids(pm, nw, :bus)
    )

    if bounded
        for (i,bus) in _PMD.ref(pm, nw, :bus)
            for (idx, t) in enumerate(terminals[i])
                if haskey(bus, "vamin")
                    _PMD.set_lower_bound(va[i][t], bus["vamin"][idx])
                    @warn " va min bounds defined "
                end
                if haskey(bus, "vamax")
                    _PMD.set_upper_bound(va[i][t], bus["vamax"][idx])
                    @warn " va max bounds defined "
                end
            end
        end
    end

    report && _IM.sol_component_value(pm, _PMD.pmd_it_sym, nw, :bus, :va, _PMD.ids(pm, nw, :bus), va)
end

## ACR

""
function variable_mc_bus_voltage(pm::_PMD.AbstractUnbalancedACRModel; nw::Int=_PMD.nw_id_default, bounded::Bool=true, report::Bool=true)
    variable_mc_bus_voltage_real(pm; nw=nw, bounded=bounded, report=report)
    variable_mc_bus_voltage_imaginary(pm; nw=nw, bounded=bounded, report=report)

    # local infeasbility issues without proper initialization;
    # convergence issues start when the equivalent angles of the starting point
    # are further away than 90 degrees from the solution (as given by ACP)
    # this is the default behaviour of _PM, initialize all phases as (1,0)
    # the magnitude seems to have little effect on the convergence (>0.05)
    # updating the starting point to a balanced phasor does the job
    for id in _PMD.ids(pm, nw, :bus)
        busref = _PMD.ref(pm, nw, :bus, id)
        terminals = busref["terminals"]
        grounded = busref["grounded"]

        ncnd = length(terminals)

        if haskey(busref, "vr_start") && haskey(busref, "vi_start")
            vr = busref["vr_start"]
            vi = busref["vi_start"]
        else
            vm_start = fill(1.0, 3)
            for t in 1:3
                if t in terminals
                    vmax = busref["vmax"][findfirst(isequal(t), terminals)]
                    vm_start[t] = min(vm_start[t], vmax)

                    vmin = busref["vmin"][findfirst(isequal(t), terminals)]
                    vm_start[t] = max(vm_start[t], vmin)
                end
            end

            vm = haskey(busref, "vm_start") ? busref["vm_start"] : haskey(busref, "vm") ? busref["vm"] : [vm_start..., fill(0.0, ncnd)...][terminals]
            va = haskey(busref, "va_start") ? busref["va_start"] : haskey(busref, "va") ? busref["va"] : [deg2rad.([0, -120, 120])..., zeros(length(terminals))...][terminals]

            vr = vm .* cos.(va)
            vi = vm .* sin.(va)
        end

        for (idx,t) in enumerate(terminals)
            JuMP.set_start_value(_PMD.var(pm, nw, :vr, id)[t], vr[idx])
            JuMP.set_start_value(_PMD.var(pm, nw, :vi, id)[t], vi[idx])
        end
    end

    # apply bounds if bounded
    if bounded
        for i in _PMD.ids(pm, nw, :bus)
            _PMD.constraint_mc_voltage_magnitude_bounds(pm, i; nw=nw)
            constraint_mc_voltage_angle_bounds(pm, i; nw=nw)
        end
    end
end


""
function variable_mc_bus_voltage_real(pm::_PMD.AbstractUnbalancedACRModel; nw::Int=_PMD.nw_id_default, bounded::Bool=true, report::Bool=true)
    terminals = Dict(i => bus["terminals"] for (i, bus) in _PMD.ref(pm, nw, :bus))

    vr = _PMD.var(pm, nw)[:vr] = Dict(i => JuMP.@variable(pm.model,
            [t in terminals[i]], base_name="$(nw)_vr_$(i)",
            start = _PMD.comp_start_value(_PMD.ref(pm, nw, :bus, i), "vr_start", t, 1.0)
        ) for i in _PMD.ids(pm, nw, :bus)
    )
    report && _IM.sol_component_value(pm, _PMD.pmd_it_sym, nw, :bus, :vr, _PMD.ids(pm, nw, :bus), vr)
end


""
function variable_mc_bus_voltage_imaginary(pm::_PMD.AbstractUnbalancedACRModel; nw::Int=_PMD.nw_id_default, bounded::Bool=true, report::Bool=true)
    terminals = Dict(i => bus["terminals"] for (i,bus) in _PMD.ref(pm, nw, :bus))
    vi = _PMD.var(pm, nw)[:vi] = Dict(i => JuMP.@variable(pm.model,
    [t in terminals[i]], base_name="$(nw)_vi_$(i)",
    start = _PMD.comp_start_value(_PMD.ref(pm, nw, :bus, i), "vi_start", t, 0.0)
    ) for i in _PMD.ids(pm, nw, :bus)
    )
    report && _IM.sol_component_value(pm, _PMD.pmd_it_sym, nw, :bus, :vi, _PMD.ids(pm, nw, :bus), vi)
end

function constraint_mc_voltage_angle_bounds(pm::_PMD.AbstractUnbalancedACRModel, i::Int; nw::Int=_PMD.nw_id_default)::Nothing
    bus = _PMD.ref(pm, nw, :bus, i)
    terminals = length(bus["terminals"])
    if haskey(bus, "vamin") && haskey(bus, "vamax")    
        va = haskey(bus, "va_start") ? bus["va_start"] : haskey(bus, "va") ? bus["va"] : [deg2rad.([0, -120, 120])..., zeros(length(terminals))...][terminals]
        println("there is va")
        @warn va
        vamin = _PMD.get(bus, "vamin", fill(0.0, length(bus["terminals"])))
        vamax = _PMD.get(bus, "vamax", fill(Inf, length(bus["terminals"])))
        println("there is vamin")
        @warn vamin
        println("there is vamax")
        @warn vamax
        constraint_mc_voltage_angle_bounds(pm, nw, i, vamin, vamax)
    end
    nothing
end

"`vamin <= va[i] <= vamax`"
function constraint_mc_voltage_angle_bounds(pm::_PMD.AbstractUnbalancedACRModel, nw::Int, i::Int, vamin::Vector{<:Real}, vamax::Vector{<:Real})
    @assert all(vamin .<= vamax)
    vr = _PMD.var(pm, nw, :vr, i)
    vi = _PMD.var(pm, nw, :vi, i)
    bus = _PMD.ref(pm, nw, :bus, i)
    grounded = bus["grounded"]
    proposed =  haskey(bus, "proposed") ? bus["proposed"] : false
    @show proposed

    proposed ? constraint_mc_voltage_phasor_angle_difference(pm, i;  min_angle_diff_deg = 100, max_angle_diff_deg = 140) : nothing

    if _N_IDX in _PMD.ref(pm, nw, :bus, i)["terminals"]         
        for (idx,t) in enumerate(_PMD.ref(pm, nw, :bus, i)["terminals"])
            
            if vamin[idx] > -Inf
                a= JuMP.@constraint(pm.model, tan(vamin[idx]) <= (vi[t]- vi[_N_IDX])/(vr[t]-vr[_N_IDX]))
                @info " \n consrtained minimum angle vamin  terminal $(idx) of bus $i:: \n  $a" 
            end

            if vamax[idx] < Inf
                b= JuMP.@constraint(pm.model, tan(vamax[idx]) >= (vi[t]- vi[_N_IDX])/(vr[t]-vr[_N_IDX]))
                @info " \n consrtained maximum angle vamax  terminal $(idx) of bus $i:: \n $b"
            end
            
            if (vamax[idx] == Inf || vamin[idx] == -Inf) && grounded[idx] == 1  && proposed # && idx == _N_IDX
                c= JuMP.@constraint(pm.model, vr[t] == 0.0 )
                d= JuMP.@constraint(pm.model, vi[t] == 0.0 )
                @info " \n consrtained neutral grounding :: \n $c \n and $d \n"

            end
        end
        
    
    else # Kron-reduced 

        for (idx,t) in enumerate(_PMD.ref(pm, nw, :bus, i)["terminals"])
            if vamin[idx] > -Inf
                JuMP.@constraint(pm.model, tan(vamin[idx]) <= vi[t]/vr[t])
                #JuMP.@constraint(pm.model, vamin[idx] <= atan(vi[t],vr[t]))
                @warn "consrtained minimum angle vamin" 
            end
            if vamax[idx] < Inf
                JuMP.@constraint(pm.model, tan(vamax[idx]) >= vi[t]/vr[t])
                #JuMP.@constraint(pm.model, vamax[idx] >= atan(vi[t],vr[t]))
                @warn "consrtained maximum angle vamax "
            end
        end
    end
end



"""
    constraint_mc_voltage_phasor_angle_difference(
        pm::_PMD.AbstractUnbalancedIVRModel, 
        i::Int; 
        nw::Int=_IM.nw_id_default, 
        min_angle_diff_deg::Real, 
        max_angle_diff_deg::Real
    )

Ensures that the angle difference between any two phase voltage phasors at bus `i`
is between `min_angle_diff_deg` and `max_angle_diff_deg`. #TODO: pass as dict items
Assumes `0 <= min_angle_diff_deg <= max_angle_diff_deg <= 180`.
The constraint derived is (P - CL*sqrt(vma*vmb))*(P - CU*sqrt(vma*vmb)) <= 0,
where P = vr1*vr2 + vi1*vi2, vma = vr1^2+vi1^2, vmb = vr2^2+vi2^2,
CL = cos(max_angle_diff_rad), CU = cos(min_angle_diff_rad).
"""
function constraint_mc_voltage_phasor_angle_difference(
    pm::_PMD.AbstractUnbalancedIVRModel, 
    i::Int; 
    nw::Int=_IM.nw_id_default, 
    min_angle_diff_deg=90::Real, 
    max_angle_diff_deg=150::Real
)
    bus = _PMD.ref(pm, nw, :bus, i)
    terminals = bus["terminals"]
    
    terminals = collect(setdiff(terminals, [_N_IDX]))

    if length(terminals) < 2
        # this only applied for three-phase connections
        return
    end

    if !(0.0 <= min_angle_diff_deg <= max_angle_diff_deg <= 180.0)
        # Make sure the assumption holds (it should always hold as the difference shouldn't be too big or identical)
        error("Angle limits [$min_angle_diff_deg, $max_angle_diff_deg] are not valid. Ensure 0 <= min <= max <= 180.")
    end

    min_angle_diff_rad = deg2rad(min_angle_diff_deg)
    max_angle_diff_rad = deg2rad(max_angle_diff_deg)

    # CU is cos of the smaller angle, CL is cos of the larger angle
    # Since cos is decreasing on [0, pi], cos(min_angle) >= cos(max_angle)
    cos_upper_bound = cos(min_angle_diff_rad) # CU
    cos_lower_bound = cos(max_angle_diff_rad) # CL

    vr = _PMD.var(pm, nw, :vr, i)
    vi = _PMD.var(pm, nw, :vi, i)

    for idx1 in axes(terminals, 1)
        for idx2 in (idx1+1):last(axes(terminals, 1))
            t1 = terminals[idx1]
            t2 = terminals[idx2]

            vra = vr[t1]
            via = vi[t1]
            vrb = vr[t2]
            vib = vi[t2]

            JuMP.@constraint(pm.model, 
                (vra * vrb + via * vib)^2 - 
                (vra * vrb + via * vib) * (cos_lower_bound + cos_upper_bound) * 
                    sqrt( (vra^2 + via^2) * (vrb^2 + vib^2) ) + 
                cos_lower_bound * cos_upper_bound * (vra^2 + via^2) * (vrb^2 + vib^2) <= 0.0
            )
            @info "constrained voltage phasor diff at bus $i between terminal $idx1 and terminal $idx2"
        end
    end
end

# written but temporary not used to control the flow for other case studies #TODO: make it done by default in the explicit neutral formulation
function constraint_mc_neutral_grounding(pm::_PMD.AbstractUnbalancedPowerModel, i::Int; nw::Int=_PMD.nw_id_default)
    vr = _PMD.var(pm, nw, :vr, i)
    vi = _PMD.var(pm, nw, :vi, i)
    bus = _PMD.ref(pm, nw, :bus, i)
    terminals = bus["terminals"]
    grounded = bus["grounded"]
    
    for (idx, t) in enumerate(terminals)
        if grounded[idx] == 1
            JuMP.@constraint(pm.model, vr[t] == 0)
            JuMP.@constraint(pm.model, vi[t] == 0)
            @debug "Constrained grounding at reference bus $i" 
        end
    end

    # if grounded[_N_IDX] == 1
    #        JuMP.@constraint(pm.model, vr[_N_IDX] == 0)
    #        JuMP.@constraint(pm.model, vi[_N_IDX] == 0)
    # end
end


#    Explicit Neutral related Constraints
###########################
"""
	function variable_mc_bus_voltage(
		pm::RectangularVoltageExplicitNeutralModels;
		nw=_PMD.nw_id_default,
		bounded::Bool=true,
	)

Creates rectangular voltage variables `:vr` and `:vi` for models with explicit neutrals
"""
function variable_mc_bus_voltage(pm::_PMD.IVRENPowerModel; nw::Int=_PMD.nw_id_default, bounded::Bool=true, report::Bool=true)
    _PMD.variable_mc_bus_voltage_real(pm; nw=nw, bounded=bounded, report=report)
    _PMD.variable_mc_bus_voltage_imaginary(pm; nw=nw, bounded=bounded, report=report)
    # apply bounds if bounded
    if bounded
        for i in _PMD.ids(pm, nw, :bus)
            _PMD.constraint_mc_voltage_magnitude_bounds(pm, i; nw=nw)
            constraint_mc_voltage_angle_bounds(pm, i; nw=nw)
        end
    end
end



"""
	function constraint_mc_generator_current_se(
		pm::AbstractExplicitNeutralIVRModel,
		id::Int;
		nw::Int=nw_id_default,
		report::Bool=true,
		bounded::Bool=true
	)

For IVR models with explicit neutrals,
creates expressions for the terminal current flows `:crg_bus` and `:cig_bus`.
"""
function constraint_mc_generator_current_se(pm::_PMD.IVRENPowerModel, id::Int; nw::Int=_PMD.nw_id_default, report::Bool=true, bounded::Bool=true)
    generator = _PMD.ref(pm, nw, :gen, id)

    nphases = _PMD._infer_int_dim_unit(generator, false)
    # Note that one-dimensional delta generators are handled as wye-connected generators.
    # The distinction between one-dimensional wye and delta generators is purely semantic
    # when neutrals are modeled explicitly.
    if get(generator, "configuration", _PMD.WYE) == _PMD.WYE || nphases==1
        constraint_mc_generator_current_wye_se(pm, nw, id, generator["connections"]; report=report, bounded=bounded)
    else
        constraint_mc_generator_current_delta_se(pm, nw, id, generator["connections"]; report=report, bounded=bounded)
    end
end

"""
	function constraint_mc_generator_current_wye_se(
		pm::_PMD.IVRENPowerModel,
		nw::Int,
		id::Int,
		connections::Vector{Int};
		report::Bool=true,
		bounded::Bool=true
	)

For IVR models with explicit neutrals,
creates expressions for the terminal current flows `:crg_bus` and `:cig_bus` of wye-connected generators
"""
function constraint_mc_generator_current_wye_se(pm::_PMD.IVRENPowerModel, nw::Int, id::Int, connections::Vector{Int}; report::Bool=true, bounded::Bool=true)
    crg = _PMD.var(pm, nw, :crg, id)
    cig = _PMD.var(pm, nw, :cig, id)
    _PMD.var(pm, nw, :crg_bus)[id] = _PMD._merge_bus_flows(pm, [crg..., -sum(crg)], connections)
    _PMD.var(pm, nw, :cig_bus)[id] = _PMD._merge_bus_flows(pm, [cig..., -sum(cig)], connections)
end

"""
	function constraint_mc_generator_current_delta_se(
		pm::_PMD.IVRENPowerModel,
		nw::Int,
		id::Int,
		connections::Vector{Int};
		report::Bool=true,
		bounded::Bool=true
	)

For IVR models with explicit neutrals,
creates expressions for the terminal current flows `:crg_bus` and `:cig_bus` of delta-connected generators
"""
function constraint_mc_generator_current_delta_se(pm::_PMD.IVRENPowerModel, nw::Int, id::Int, connections::Vector{Int}; report::Bool=true, bounded::Bool=true)
    crg = _PMD.var(pm, nw, :crg, id)
    cig = _PMD.var(pm, nw, :cig, id)
    Md = _PMD._get_delta_transformation_matrix(length(connections))
    _PMD.var(pm, nw, :crg_bus)[id] = _PMD._merge_bus_flows(pm, Md'*crg, connections)
    _PMD.var(pm, nw, :cig_bus)[id] = _PMD._merge_bus_flows(pm, Md'*cig, connections)
end





function constraint_mc_current_balance_se(pm::_PMD.IVRENPowerModel, nw::Int, i::Int, terminals::Vector{Int}, grounded::Vector{Bool}, bus_arcs::Vector{Tuple{Tuple{Int,Int,Int},Vector{Int}}}, bus_arcs_sw::Vector{Tuple{Tuple{Int,Int,Int},Vector{Int}}}, bus_arcs_trans::Vector{Tuple{Tuple{Int,Int,Int},Vector{Int}}}, bus_gens::Vector{Tuple{Int,Vector{Int}}}, bus_storage::Vector{Tuple{Int,Vector{Int}}}, bus_loads::Vector{Tuple{Int,Vector{Int}}}, bus_shunts::Vector{Tuple{Int,Vector{Int}}})
    #NB only difference with pmd is crd_bus replaced by crd, and same with cid
    vr = _PMD.var(pm, nw, :vr, i)
    vi = _PMD.var(pm, nw, :vi, i)
    cr    = get(_PMD.var(pm, nw),    :cr_bus, Dict()); _PMD._check_var_keys(cr, bus_arcs, "real current", "branch")
    ci    = get(_PMD.var(pm, nw),    :ci_bus, Dict()); _PMD._check_var_keys(ci, bus_arcs, "imaginary current", "branch")
    crd   = get(_PMD.var(pm, nw),   :crd, Dict()); _PMD._check_var_keys(crd, bus_loads, "real current", "load")
    cid   = get(_PMD.var(pm, nw),   :cid, Dict()); _PMD._check_var_keys(cid, bus_loads, "imaginary current", "load")
    crg   = get(_PMD.var(pm, nw),   :crg_bus, Dict()); _PMD._check_var_keys(crg, bus_gens, "real current", "generator")
    cig   = get(_PMD.var(pm, nw),   :cig_bus, Dict()); _PMD._check_var_keys(cig, bus_gens, "imaginary current", "generator")
    crs   = get(_PMD.var(pm, nw),   :crs, Dict()); _PMD._check_var_keys(crs, bus_storage, "real currentr", "storage")
    cis   = get(_PMD.var(pm, nw),   :cis, Dict()); _PMD._check_var_keys(cis, bus_storage, "imaginary current", "storage")
    crsw  = get(_PMD.var(pm, nw),  :crsw, Dict()); _PMD._check_var_keys(crsw, bus_arcs_sw, "real current", "switch")
    cisw  = get(_PMD.var(pm, nw),  :cisw, Dict()); _PMD._check_var_keys(cisw, bus_arcs_sw, "imaginary current", "switch")
    crt   = get(_PMD.var(pm, nw),   :crt, Dict()); _PMD._check_var_keys(crt, bus_arcs_trans, "real current", "transformer")
    cit   = get(_PMD.var(pm, nw),   :cit, Dict()); _PMD._check_var_keys(cit, bus_arcs_trans, "imaginary current", "transformer")

    Gs, Bs = _PMD._build_bus_shunt_matrices(pm, nw, terminals, bus_shunts)

    ungrounded_terminals = [(idx,t) for (idx,t) in enumerate(terminals) if !grounded[idx]]
    #ungrounded_terminals = [(idx,t) for (idx,t) in enumerate(terminals)]
    
    for (idx, t) in ungrounded_terminals      
        kcl_real_part = JuMP.@constraint(pm.model,  sum(cr[a][t] for (a, conns) in bus_arcs if t in conns)
                                    + sum(crsw[a_sw][t] for (a_sw, conns) in bus_arcs_sw if t in conns)
                                    + sum(crt[a_trans][t] for (a_trans, conns) in bus_arcs_trans if t in conns)
                                    ==
                                      sum(crg[g][t]         for (g, conns) in bus_gens if t in conns)
                                    - sum(crs[s][t]         for (s, conns) in bus_storage if t in conns)
                                    - sum(crd[d][t]         for (d, conns) in bus_loads if t in conns)
                                    - sum( Gs[idx,jdx]*vr[u] -Bs[idx,jdx]*vi[u] for (jdx,u) in ungrounded_terminals) # shunts
                                    )
        #println("The KCL of Ireal across bus $(i) at terminal $(t) is (which is surrounded by $(length(bus_arcs)) branches, $(length(bus_loads)) loads, and $(length(bus_gens)) generators): ")
        #display(kcl_real_part)
        kcl_imaginary_part = JuMP.@constraint(pm.model, sum(ci[a][t] for (a, conns) in bus_arcs if t in conns)
                                    + sum(cisw[a_sw][t] for (a_sw, conns) in bus_arcs_sw if t in conns)
                                    + sum(cit[a_trans][t] for (a_trans, conns) in bus_arcs_trans if t in conns)
                                    ==
                                      sum(cig[g][t]         for (g, conns) in bus_gens if t in conns)
                                    - sum(cis[s][t]         for (s, conns) in bus_storage if t in conns)
                                    - sum(cid[d][t]         for (d, conns) in bus_loads if t in conns)
                                    - sum( Gs[idx,jdx]*vi[u] +Bs[idx,jdx]*vr[u] for (jdx,u) in ungrounded_terminals) # shunts
                                    )
        #println("The KCL Iimaginary across bus $(i) at terminal $(t) is (which is surrounded by $(length(bus_arcs)) branches, $(length(bus_loads)) loads, and $(length(bus_gens)) generators): ")
        #display(kcl_imaginary_part)
    end
end
