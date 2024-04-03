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
function constraint_mc_theta_ref(pm::_PMD.AbstractUnbalancedPowerModel, i::Int; nw::Int=_PMD.nw_id_default)::Nothing
    bus = _PMD.ref(pm, nw, :bus, i)
    terminals = bus["terminals"]
    if haskey(bus, "va")
        va_ref = _PMD.get( _PMD.ref(pm, nw, :bus, i), "va", [deg2rad.([0.0, -120.0, 120.0])..., zeros(length(terminals))...][terminals])
        display("the va_ref is moved as: $va_ref")
        constraint_mc_theta_ref(pm, nw, i, va_ref)
    end
    nothing
end



"Creates phase angle constraints at reference buses"
function constraint_mc_theta_ref(pm::_PMD.AbstractUnbalancedPolarModels, nw::Int, i::Int, va_ref::Vector{<:Real})
    terminals = _PMD.ref(pm, nw, :bus, i)["terminals"]
   
    if va_ref == deg2rad.([0.0, -1, -1]) #in case only one angle satisfied
        va = _PMD.var(pm, nw, :va, i)          
        display(" PhA only - constraint defined with: $va_ref[1]")
        JuMP.@constraint(pm.model, va[1] == va_ref[1]) #can be replaced with generic bounds

    elseif va_ref == deg2rad.([0.0, -2, -2]) #in case all angles satisfied
        va = _PMD.var(pm, nw, :va, i)          
        display(" PhA only (PhB,C bounded) - constraint defined with: $va_ref[1]")
        JuMP.@constraint(pm.model, va[1] == va_ref[1]) #can be replaced with generic bounds
        ϵ = (5 * π/180)
        #setting upper and lower bounds for the phase angle
        JuMP.@constraint(pm.model, (-2π/3 - ϵ) <= va[2] <= (-2π/3 + ϵ))
        JuMP.@constraint(pm.model, (2π/3 - ϵ) <= va[3] <= (2π/3 + ϵ))
        display(" constrained: $va with ϵ = $ϵ, ph.c bounds:$((-2π/3 - ϵ)) and  $((-2π/3 + ϵ))  ph.b bounds: $((2π/3 - ϵ)) and  $((2π/3 + ϵ))") 

    
        
    else     
        va = [_PMD.var(pm, nw, :va, i)[t] for t in terminals]
        display(" PhA, PhB, PhC - constraint defined with: $va_ref")
        JuMP.@constraint(pm.model, va .== va_ref)
    end
end

"defined minimum and maximum voltage magnitude bounds for each terminal of the bus"
function constraint_mc_voltage_bounds_se(pm::_PMD.AbstractUnbalancedPolarModels, nw::Int, i::Int)
    bus = _PMD.ref(pm, nw, :bus, i)
    terminals = bus["terminals"]
    vm = _PMD.var(pm, nw, :vm, i)

    vm_max = _PMD.get( _PMD.ref(pm, nw, :bus, i), "vm_max", [[1.2, 1.2, 1.2]..., zeros(length(terminals))...][terminals])
    vm_min = _PMD.get( _PMD.ref(pm, nw, :bus, i), "vm_min", [[0.8, 0.8, 0.8]..., zeros(length(terminals))...][terminals])
    
    for t in terminals
        JuMP.@constraint(pm.model, vm_min[t] <= vm[t] <= vm_max[t])
    end
    display(" constrained: $vm with vm_min: $vm_min and vm_max: $vm_max")

end