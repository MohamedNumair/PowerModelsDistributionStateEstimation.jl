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
    conns = get_active_connections(pm, nw, _PMD.ref(pm, nw, :meas, i, "cmp"), cmp_id)

    for (idx, c) in enumerate(conns)
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
            JuMP.@NLconstraint(pm.model,
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

"""
    constraint_mc_theta_ref(pm::AbstractUnbalancedPowerModel, i::Int; nw::Int=nw_id_default)::Nothing

function for (unconstrained) reference angle with relaxed constraints.
"""
# function constraint_mc_theta_ref(pm::_PMD.AbstractUnbalancedPowerModel, i::Int; nw::Int=_PMD.nw_id_default)::Nothing
#     bus = _PMD.ref(pm, nw, :bus, i)
#     terminals = bus["terminals"]
#     if haskey(bus, "va")
#         va_ref = _PMD.get( _PMD.ref(pm, nw, :bus, i), "va", [deg2rad.([0.0, -120.0, 120.0])..., zeros(length(terminals))...][terminals])
#         display("the va_ref is moved as: $va_ref")
#         constraint_mc_theta_ref(pm, nw, i, va_ref)
#     end
#     nothing
# end





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
                end
                if haskey(bus, "vamax")
                    _PMD.set_upper_bound(va[i][t], bus["vamax"][idx])
                end
            end
        end
        display(" va bounds defined ")
    end


    report && _IM.sol_component_value(pm, _PMD.pmd_it_sym, nw, :bus, :va, _PMD.ids(pm, nw, :bus), va)
end


""
function variable_mc_bus_voltage_magnitude_only(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_PMD.nw_id_default, bounded::Bool=true, report::Bool=true)
    terminals = Dict(i => bus["terminals"] for (i,bus) in _PMD.ref(pm, nw, :bus))
    vm = _PMD.var(pm, nw)[:vm] = Dict(i => JuMP.@variable(pm.model,
            [t in terminals[i]], base_name="$(nw)_vm_$(i)",
            start = _PMD.comp_start_value(_PMD.ref(pm, nw, :bus, i), ["vm_start", "vm"], t, 1.0)
        ) for i in _PMD.ids(pm, nw, :bus)
    )

    if bounded
        for (i,bus) in _PMD.ref(pm, nw, :bus)
            for (idx, t) in enumerate(terminals[i])
                if haskey(bus, "vmin")
                    _PMD.set_lower_bound(vm[i][t], bus["vmin"][idx])
                end
                if haskey(bus, "vmax")
                    _PMD.set_upper_bound(vm[i][t], bus["vmax"][idx])
                end
            end
        end
        display(" vm bounds defined ")
    end

    report && _IM.sol_component_value(pm, _PMD.pmd_it_sym, nw, :bus, :vm, _PMD.ids(pm, nw, :bus), vm)
end







# "Creates phase angle constraints at _PMD.reference buses"
# function constraint_mc_theta_ref_pha(pm::_PMD.AbstractUnbalancedPolarModels, nw::Int, i::Int, va_ref::Vector{<:Real})
#     terminals = _PMD.ref(pm, nw, :bus, i)["terminals"] 
#     va = _PMD.var(pm, nw, :va, i)          
#     display(" Constraining Reference Bus's Phase (a) angles: $va_ref[1]")
#     JuMP.@constraint(pm.model, va[1] == va_ref[1]) #can be replaced with generic bounds     
# end


#" bounds the angles of all terminals of the buses"

# function constraint_mc_theta_bounds(pm::_PMD.AbstractUnbalancedPowerModel, i::Int; nw::Int=_PMD.nw_id_default)
#     bus = _PMD.ref(pm, nw, :bus, i)
#     terminals = bus["terminals"]
#     va = _PMD.var(pm, nw, :va, i)
#     display(" Bounding all angles of all buses: $i")
#     # JuMP.@constraint(pm.model, va[1] == va_ref[1]) #can be replaced with generic bounds
#     # ϵ = (0.01 * π/180)
#     # #setting upper and lower bounds for the phase angle
#     # JuMP.@constraint(pm.model, (-2π/3 - ϵ) <= va[2] <= (-2π/3 + ϵ))
#     # JuMP.@constraint(pm.model, (2π/3 - ϵ) <= va[3] <= (2π/3 + ϵ))
#     va_max = _PMD.get( _PMD.ref(pm, nw, :bus, i), "va_max", [deg2rad.([0.0, -120.0, 120.0])..., zeros(length(terminals))...][terminals])
#     va_min = _PMD.get( _PMD.ref(pm, nw, :bus, i), "va_min", [deg2rad.([0.0, -120.0, 120.0])..., zeros(length(terminals))...][terminals])

#     display(" va bounds defined with: $va_max and $va_min")
#     for t in terminals
#         JuMP.@constraint(pm.model, va_min[t] <= va[t] <= va_max[t])
#     end
#     #display(" va bounds constrained: $va with va_min: $va_min and va_max: $va_max")
# end


# "defined minimum and maximum voltage magnitude bounds for each terminal of the bus"
# function constraint_mc_voltage_bounds_se(pm::_PMD.AbstractUnbalancedPowerModel, i::Int; nw::Int=_PMD.nw_id_default)
#     bus = _PMD.ref(pm, nw, :bus, i)
#     terminals = bus["terminals"]
#     vm = _PMD.var(pm, nw, :vm, i)
#     vm_ub = 1.4 
#     vm_lb = 0.6
#     vm_max = _PMD.get( _PMD.ref(pm, nw, :bus, i), "vm_max", [[vm_ub, vm_ub, vm_ub]..., zeros(length(terminals))...][terminals])
#     vm_min = _PMD.get( _PMD.ref(pm, nw, :bus, i), "vm_min", [[vm_lb, vm_lb, vm_lb]..., zeros(length(terminals))...][terminals])
#     display(" vm bounds defined with: $vm_max and $vm_min")
#     for t in terminals
#         JuMP.@constraint(pm.model, vm_min[t] <= vm[t] <= vm_max[t])
#     end
#     #display(" vm bounds constrained: $vm with vm_min: $vm_min and vm_max: $vm_max")

# end

